"""
MIT License

Copyright (c) 2017 Grant Van Horn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
# pylint: disable=line-too-long

from __future__ import absolute_import, division, print_function

from collections import Counter
import math
import random

import numpy as np

from ...crowdsourcing import CrowdDataset, CrowdImage, CrowdWorker, CrowdLabel
from ...util.taxonomy import Taxonomy

class CrowdDatasetMulticlassSingleBinomial(CrowdDataset):
    """ A dataset for multiclass labeling using a single binomial skill
    parameter at each inner node of the taxonomy.
    """

    def __init__(self,

                 # Taxonomy of label nodes
                 taxonomy=None,

                 # Prior probability on the classes: p(y)
                 # An iterable, {class_key : probability of class}
                 class_probs=None,
                 # Global priors for the probability of a class occuring,
                 # used to estimate the class priors
                 class_probs_prior_beta=10,
                 # An iterable, {class_key : prior probability of class}
                 class_probs_prior=None,

                 # Global priors used to compute the pooled probabilities for a
                 # worker being correct
                 prob_correct_prior_beta=15,
                 prob_correct_prior=0.8,
                 prob_correct_beta=10,

                 # Global priors used to compute worker trust
                 prob_trust_prior_beta=15,
                 prob_trust_prior=0.8,
                 prob_trust_beta=10,
                 prob_trust=0.8,

                 # TODO: refactor to `verification_task`
                 model_worker_trust=False,
                 # TODO: refactor to `dependent_verification`
                 recursive_trust=False,
                 **kwargs):

        super(CrowdDatasetMulticlassSingleBinomial, self).__init__(**kwargs)
        self._CrowdImageClass_ = CrowdImageMulticlassSingleBinomial
        self._CrowdWorkerClass_ = CrowdWorkerMulticlassSingleBinomial
        self._CrowdLabelClass_ = CrowdLabelMulticlassSingleBinomial

        self.class_probs = class_probs
        self.class_probs_prior_beta = class_probs_prior_beta
        self.class_probs_prior = class_probs_prior

        # These are the dataset wide priors that we use to estimate the per
        # worker skill parameters
        self.taxonomy = taxonomy
        self.prob_correct_prior_beta = prob_correct_prior_beta
        self.prob_correct_prior = prob_correct_prior
        self.prob_correct_beta = prob_correct_beta

        # Worker trust probabilities
        self.prob_trust_prior_beta = prob_trust_prior_beta
        self.prob_trust_prior = prob_trust_prior
        self.prob_trust_beta = prob_trust_beta
        self.prob_trust = prob_trust

        self.model_worker_trust = model_worker_trust
        self.recursive_trust = recursive_trust



        # NOTE: not sure what to do here, one for each class?
        # How should it be plotted?
        self.skill_names = ['Prob Correct']
        if model_worker_trust:
            self.skill_names.append('Prob Trust')

        self.encode_exclude['taxonomy'] = True

    def copy_parameters_from(self, dataset, full=True):
        super(CrowdDatasetMulticlassSingleBinomial, self).copy_parameters_from(
            dataset,
            full=full
        )

        self.class_probs = dataset.class_probs
        self.class_probs_prior_beta = dataset.class_probs_prior_beta
        self.class_probs_prior = dataset.class_probs_prior

        self.taxonomy = dataset.taxonomy.duplicate(duplicate_data=True)
        self.taxonomy.finalize()
        if hasattr(dataset.taxonomy, 'priors_initialized'):
            self.taxonomy.priors_initialized = \
                dataset.taxonomy.priors_initialized

        self.prob_correct_prior_beta = dataset.prob_correct_prior_beta
        self.prob_correct_prior = dataset.prob_correct_prior

        self.prob_correct_beta = dataset.prob_correct_beta

        self.prob_trust_prior_beta = dataset.prob_trust_prior_beta
        self.prob_trust_prior = dataset.prob_trust_prior
        self.prob_trust_beta = dataset.prob_trust_beta
        self.prob_trust = dataset.prob_trust

        self.model_worker_trust = dataset.model_worker_trust
        self.recursive_trust = dataset.recursive_trust

        self.estimate_priors_automatically = \
            dataset.estimate_priors_automatically

    def initialize_default_priors(self):
        """ Convenience function for initializing all the priors to default
        values.
        """
        # All nodes will be initialized to have `prob_correct_prior`
        for node in self.taxonomy.breadth_first_traversal():
            node.data['prob'] = 0.
            if not node.is_leaf:
                node.data['prob_correct_prior'] = self.prob_correct_prior
                node.data['prob_correct'] = self.prob_correct_prior

        # Initialize the node probabilities (of occuring)
        for leaf_node in self.taxonomy.leaf_nodes():
            prob_y = self.class_probs[leaf_node.key]
            leaf_node.data['prob'] = prob_y
            # Update the node distributions
            for ancestor in leaf_node.ancestors:
                ancestor.data['prob'] += prob_y

        self.taxonomy.priors_initialized = True

    def estimate_priors(self, gt_dataset=None):
        """Estimate the dataset-wide parameters.
        For the full dataset (given a gt_dataset) we want to estimate the class
        priors.
        """

        # Initialize the `prob_correct_prior` for each node to
        # `self.prob_correct_prior`
        if not self.taxonomy.priors_initialized:
            print("INITIALIZING all node priors to defaults")
            self.initialize_default_priors()
            self.taxonomy.priors_initialized = True

        # Pooled counts
        for node in self.taxonomy.breadth_first_traversal():
            if not node.is_leaf:
                # [num, denom] => [# correct, # total]
                node.data['prob_correct_counts'] = [0, 0]
                node.data['prob'] = 0

        # Counts for the classes
        class_dist = {node.key: 0. for node in self.taxonomy.leaf_nodes()}

        # Go through each image and add to the counts
        for i in self.images:

            # Does this image have a computer vision annotation?
            has_cv = 0
            if self.cv_worker and self.cv_worker.id in self.images[i].z:
                has_cv = 1

            # Skip this image if it doesn't have at least human annotations.
            if len(self.images[i].z) - has_cv <= 1:
                continue

            # If we have access to a ground truth dataset, then use the label
            # from there.
            if gt_dataset is not None:
                y = gt_dataset.images[i].y.label
            # Otherwise, grab the current prediction for the image
            else:
                y = self.images[i].y.label

            # Update the class distributions
            class_dist[y] += 1.

            y_node = self.taxonomy.nodes[y]
            y_level = y_node.level

            # Go through each worker and add their annotation to the respective
            # counts.
            for w in self.images[i].z:
                # Skip the computer vision annotations
                if not self.images[i].z[w].is_computer_vision():

                    # Worker annotation
                    z = self.images[i].z[w].label
                    z_node = self.taxonomy.nodes[z]
                    z_level = z_node.level

                    # Update the counts for each layer of the taxonomy.
                    for l in xrange(0, y_level):

                        # Get the ancestor at level `l` and the child at `l+1`
                        # for the image label
                        y_l_node = self.taxonomy.node_at_level_from_node(l, y_node)
                        y_l_child_node = self.taxonomy.node_at_level_from_node(l + 1, y_node)

                        # Update the denominator for prob_correct
                        y_l_node.data['prob_correct_counts'][1] += 1.

                        if l < z_level:

                            # Get the child at `l+1` for the worker's prediction
                            z_l_child_node = self.taxonomy.node_at_level_from_node(l + 1, z_node)

                            # Are the children nodes the same? If so then the worker
                            # was correct and we update the parent node
                            if z_l_child_node == y_l_child_node:
                                # Update the numerator for prob_correct
                                y_l_node.data['prob_correct_counts'][0] += 1.


        # compute the pooled probability of being correct priors
        for node in self.taxonomy.breadth_first_traversal():
            if not node.is_leaf:

                # Probability of predicting the children of a node correctly
                prob_correct_prior = node.data['prob_correct_prior']
                prob_correct_num = self.prob_correct_prior_beta * prob_correct_prior + node.data['prob_correct_counts'][0]
                prob_correct_denom = self.prob_correct_prior_beta + node.data['prob_correct_counts'][1]
                prob_correct_denom = np.clip(prob_correct_denom, a_min=0.00000001, a_max=None)
                node.data['prob_correct'] = np.clip(prob_correct_num / prob_correct_denom, a_min=0.00000001, a_max=0.99999)

        # Class probabilities (leaf node probabilities)
        num_images = float(np.sum(class_dist.values()))
        for y, count in class_dist.iteritems():
            num = self.class_probs_prior[y] * self.class_probs_prior_beta + count
            denom = self.class_probs_prior_beta + num_images
            self.class_probs[y] = np.clip(num / denom, a_min=0.00000001, a_max=0.999999)

        # Node probabilities:
        for leaf_node in self.taxonomy.leaf_nodes():
            prob_y = self.class_probs[leaf_node.key]
            leaf_node.data['prob'] = prob_y
            # Update the node distributions
            for ancestor in leaf_node.ancestors:
                ancestor.data['prob'] += prob_y

        # Probability of a worker trusting previous annotations
        # (with a Beta prior)
        if self.model_worker_trust:
            prob_trust_num = self.prob_trust_prior_beta * self.prob_trust_prior
            prob_trust_denom = self.prob_trust_prior_beta

            for worker_id, worker in self.workers.iteritems():
                for image in worker.images.itervalues():

                    if self.recursive_trust:
                        # Only dependent on the imediately previous value
                        worker_t = image.z.keys().index(worker_id)
                        if worker_t > 0:
                            worker_label = image.z[worker_id].label
                            prev_anno = image.z.values()[worker_t - 1]

                            prob_trust_denom += 1.
                            if worker_label == prev_anno.label:
                                prob_trust_num += 1.
                    else:
                        # Assume all of the previous labels are treated
                        # independently
                        worker_label = image.z[worker_id].label
                        for prev_worker_id, prev_anno in image.z.iteritems():
                            if prev_worker_id == worker_id:
                                break
                            if not prev_anno.is_computer_vision() or self.naive_computer_vision:
                                prob_trust_denom += 1.
                                if worker_label == prev_anno.label:
                                    prob_trust_num += 1.

            self.prob_trust = np.clip(prob_trust_num / float(prob_trust_denom), 0.00000001, 0.9999)

    def initialize_parameters(self, avoid_if_finished=False):
        """Pass on the dataset-wide worker skill priors to the workers.
        """

        for worker in self.workers.itervalues():
            if avoid_if_finished and worker.finished:
                continue

            if self.model_worker_trust:
                worker.prob_trust = self.prob_trust

            # Pass on the pooled estimates
            # Initilize the worker taxonomy if this hasn't been done
            if worker.taxonomy is None:
                worker.taxonomy = self.taxonomy.duplicate(
                    duplicate_data=True)
                worker.taxonomy.finalize()

            for node in self.taxonomy.breadth_first_traversal():
                worker_node = worker.taxonomy.nodes[node.key]
                worker_node.data['prob'] = node.data['prob']
                if not node.is_leaf:
                    worker_node.data['prob_correct'] = node.data['prob_correct']

    def parse(self, data):
        super(CrowdDatasetMulticlassSingleBinomial, self).parse(data)
        if 'taxonomy_data' in data:
            self.taxonomy = Taxonomy()
            self.taxonomy.load(data['taxonomy_data'])
            self.taxonomy.finalize()

    def encode(self):
        data = super(CrowdDatasetMulticlassSingleBinomial, self).encode()
        if self.taxonomy is not None:
            data['taxonomy_data'] = self.taxonomy.export()
        return data


class CrowdImageMulticlassSingleBinomial(CrowdImage):
    """ An image to be labeled with a multiclass label.
    """

    def __init__(self, id_, params):
        super(CrowdImageMulticlassSingleBinomial, self).__init__(id_, params)

        self.risk = 1.

    def crowdsource_simple(self, avoid_if_finished=False):
        """Simply do majority vote.
        """
        if avoid_if_finished and self.finished:
            return

        taxonomy = self.params.taxonomy
        votes = Counter([anno.label for anno in self.z.values()
                         if taxonomy.nodes[anno.label].is_leaf]).items()
        if len(votes) > 0:
            votes.sort(key=lambda x: x[1])
            votes.reverse()
            max_votes = votes[0][1]
            contenders = [x[0] for x in votes if x[1] == max_votes]

            pred_y = random.choice(contenders)
        else:
            # Randomly pick a label.
            # NOTE: could bias with the class priors.
            leaf_keys = [node.key for node in taxonomy.leaf_nodes()]
            pred_y = random.choice(leaf_keys)
        self.y = CrowdLabelMulticlassSingleBinomial(
            image=self, worker=None, label=pred_y)

    def prob_anno_given_label_and_skills(self, z_label, y_label, worker):
        """ p(z_{ij} | y_i, w_j)
        """

        taxonomy = self.params.taxonomy

        z_node = taxonomy.nodes[z_label]
        z_level = z_node.level

        y_node = taxonomy.nodes[y_label]
        y_level = y_node.level

        # Short circuit the flat list condition
        if taxonomy.max_depth == 1:
            # probability of the user selecting this node (pooled probability)
            z_node_prob = z_node.data['prob']
            w_prob_correct = worker.taxonomy.root_node.data['prob_correct']

            # accessing `key` directly to speed things up
            if y_node.key == z_node.key:
                p = w_prob_correct
            else:
                p = ((1. - w_prob_correct) * z_node_prob)
            return p

        # Taxonomic structure
        else:

            # Multiply the likelihoods from each layer of the taxonomy.
            likelihoods = np.empty([z_level])
            for l in xrange(0, z_level):

                # Get the ancestor at level `l` and the child at `l+1` for
                # the worker's prediction
                if l == 0:
                    z_l_node = taxonomy.root_node
                else:
                    z_l_node = taxonomy.node_at_level_from_node(l, z_node)
                if l + 1 == z_level:
                    z_l_child_node = z_node
                else:
                    z_l_child_node = taxonomy.node_at_level_from_node(
                        l + 1, z_node)

                # probability of the user selecting this node
                z_l_child_node_prob = z_l_child_node.data['prob']

                if l >= y_level:
                    likelihoods[l] = z_l_child_node_prob
                else:
                    if l == 0:
                        y_l_node = taxonomy.root_node
                    else:
                        y_l_node = taxonomy.node_at_level_from_node(
                            l, y_node)

                    y_l_node_prob_correct = worker.taxonomy.nodes[y_l_node.key].data['prob_correct']

                    # Are the parent nodes the same?
                    if z_l_node == y_l_node:
                        if l + 1 == y_level:
                            y_l_child_node = y_node
                        else:
                            y_l_child_node = taxonomy.node_at_level_from_node(
                                l + 1, y_node)

                        # Are the children nodes the same?
                        if y_l_child_node == z_l_child_node:
                            likelihoods[l] = y_l_node_prob_correct
                        else:
                            likelihoods[l] = (1. - y_l_node_prob_correct) * z_l_child_node_prob

                    else:
                        likelihoods[l] = (1. - y_l_node_prob_correct) * z_l_child_node_prob

            return np.prod(likelihoods)

    def predict_true_labels(self, avoid_if_finished=False):
        """ Compute the y that is most likely given the annotations, worker
        skills, etc.
        """

        if avoid_if_finished and self.finished:
            return

        taxonomy = self.params.taxonomy

        # Worker indices, most recent to oldest
        winds = self.z.keys()
        winds.reverse()
        worker_times = np.arange(len(winds))[::-1]

        # Compute the log likelihood of each class
        y_keys = np.empty(taxonomy.num_leaf_nodes, dtype=np.int)
        lls = np.empty(taxonomy.num_leaf_nodes, dtype=np.float)
        y_index = 0
        for y_node in taxonomy.leaf_nodes():
            y = y_node.key

            if self.cv_pred is not None and not self.params.naive_computer_vision:
                prob_y = self.cv_pred.prob[y]
            else:
                prob_y = self.params.class_probs[y]

            ll_y = math.log(prob_y)

            for w, worker_time in zip(winds, worker_times):
                if not self.z[w].is_computer_vision() or self.params.naive_computer_vision:

                    z = self.z[w].label

                    num = math.log(self.prob_anno_given_label_and_skills(z, y, self.z[w].worker))

                    # Are we modeling the dependence of the user labels?
                    if self.params.model_worker_trust:
                        if self.params.recursive_trust:
                            # Recursive computation
                            num += math.log(self.z[w].worker.compute_prob_of_previous_annotations(
                                self.id, z, worker_time))
                        else:
                            # Assume worker treats each previous label independently
                            prob_z = self.params.class_probs[z]
                            for prev_w in self.z:
                                if not self.z[prev_w].is_computer_vision() or self.params.naive_computer_vision:
                                    if prev_w == w:
                                        break

                                    if z == self.z[prev_w].label:
                                        num += math.log(self.z[w].worker.prob_trust)
                                    else:
                                        num += (math.log(1. - self.z[w].worker.prob_trust) + math.log(prob_z))

                        # Compute the denominator
                        denom = 0.
                        for z_other_node in taxonomy.leaf_nodes():
                            z_other = z_other_node.key
                            # Likelihood of this other label given the worker's skill
                            # p(z | y, w)
                            prob_z_other = self.prob_anno_given_label_and_skills(
                                z_other, y, self.z[w].worker)

                            # p(H^{t-1} | z, w)
                            if self.params.recursive_trust:
                                # Recursive computation
                                prob_z_other *= self.z[w].worker.compute_prob_of_previous_annotations(
                                    self.id, z_other, worker_time)
                            else:
                                # Assume worker treats each previous label independently
                                z_other_class_prob = z_other_node.data['prob']
                                for prev_w in self.z:
                                    if not self.z[prev_w].is_computer_vision() or self.params.naive_computer_vision:
                                        if prev_w == w:
                                            break
                                        if z_other == self.z[prev_w].label:
                                            prob_z_other *= self.z[w].worker.prob_trust
                                        else:
                                            prob_z_other *= ((1. - self.z[w].worker.prob_trust) * z_other_class_prob)

                            denom += prob_z_other
                        denom = math.log(denom)
                        num -= denom

                    ll_y += num

            lls[y_index] = ll_y
            y_keys[y_index] = y
            y_index += 1

        sidx = np.argsort(lls)[::-1]
        lls = lls[sidx]
        y_keys = y_keys[sidx]

        pred_y = y_keys[0]
        self.y = CrowdLabelMulticlassSingleBinomial(
            image=self, worker=None, label=pred_y)

        m = lls[0]
        num = 1.
        denom = np.sum(np.exp(lls - m))
        prob_y = num / denom
        self.risk = 1. - prob_y

    def compute_log_likelihood(self):
        """Compute the log likelihood of the predicted label given the prior
        that the class is present.
        """
        y = self.y.label

        if self.cv_pred != None:
            ll = math.log(self.cv_pred.prob[y])
        else:
            ll = math.log(self.params.class_probs[y])

        return ll

    def estimate_parameters(self, avoid_if_finished=False):
        """We didn't bother with the image difficulty parameters for this task.
        """
        return

    def check_finished(self, set_finished=True):
        """ Set finish if our risk is less than the threshold.
        """
        if self.finished:
            return True

        finished = self.risk <= self.params.min_risk
        if set_finished:
            self.finished = finished

        return finished


class CrowdWorkerMulticlassSingleBinomial(CrowdWorker):
    """ A worker providing multiclass labels.
    """
    def __init__(self, id_, params):
        super(CrowdWorkerMulticlassSingleBinomial, self).__init__(id_, params)

        # Placeholder for generic skill.
        self.prob = None
        self.skill = None

        # Copy over the global probabilities
        self.taxonomy = None
        self.encode_exclude['taxonomy'] = True

        self.prob_trust = params.prob_trust
        self._rec_cache = {}
        self.encode_exclude['_rec_cache'] = True


    def compute_log_likelihood(self):
        """ The log likelihood of the skill.
        """
        ll = 0

        for node in self.taxonomy.breadth_first_traversal():
            if not node.is_leaf:

                pooled_node = self.params.taxonomy.nodes[node.key]
                pooled_prob_correct = pooled_node.data['prob_correct']

                prob_correct = node.data['prob_correct']

                ll += ((pooled_prob_correct * self.params.prob_correct_beta - 1) * math.log(prob_correct) +
                       ((1 - pooled_prob_correct) * self.params.prob_correct_beta - 1) * math.log(1. - prob_correct))

        if self.params.model_worker_trust:
            ll += ((self.params.prob_trust * self.params.prob_trust_beta - 1) * math.log(self.prob_trust) +
                   ((1 - self.params.prob_trust) * self.params.prob_trust_beta - 1) * math.log(1. - self.prob_trust))

        return ll

    def compute_prob_of_previous_annotations(self, image_id, our_label, t,
                                             compute_denom=True):
        """ p(H^{t-1} | z_j, w_j)
        """
        prob = 1.

        # Break the recursion
        if t == 0:
            return prob

        # Check the recursion cache for precomputed values
        rec_key = (image_id, our_label, t, compute_denom)
        if rec_key in self._rec_cache:
            return self._rec_cache[rec_key]

        our_label_class_prob = self.params.class_probs[our_label]

        # probability of the previous annotation given our annotation
        prev_anno = self.images[image_id].z.values()[t - 1]
        prev_label = prev_anno.label

        if prev_label == our_label:
            prob *= self.prob_trust
        else:
            prob *= (1. - self.prob_trust) * our_label_class_prob

        prob *= prev_anno.worker.compute_prob_of_previous_annotations(
            image_id, prev_label, t - 1)

        # Compute the denominator, which is a summation of the probability of any
        # label occuring previously given our annotation
        if compute_denom:
            denom = 0.
            for z_other_node in self.taxonomy.leaf_nodes():
                z_other = z_other_node.key

                # p(z | z_j, w)
                if z_other == our_label:
                    prob_z_other = self.prob_trust
                else:
                    prob_z_other = (1. - self.prob_trust) * \
                        our_label_class_prob

                prob_z_other *= prev_anno.worker.compute_prob_of_previous_annotations(
                    image_id, z_other, t - 1)
                denom += prob_z_other
        else:
            denom = 1.

        p = prob / denom

        self._rec_cache[rec_key] = p
        return p

    def estimate_parameters(self, avoid_if_finished=False):
        """ Estimate the worker skill parameters.
        """

        assert self.taxonomy is not None, "Worker %s's taxonomy was not initialized"

        # Compute the counts for each node
        for node in self.taxonomy.breadth_first_traversal():
            if not node.is_leaf:
                node.data['prob_correct_counts'] = [0, 0]  # num, denom

        for image in self.images.itervalues():

            if len(image.z) <= 1:
                continue

            y = image.y.label
            z = image.z[self.id].label

            y_node = self.taxonomy.nodes[y]
            y_level = y_node.level

            z_node = self.taxonomy.nodes[z]
            z_level = z_node.level

            # Update the counts for each layer of the taxonomy.
            for l in range(0, y_level):
                if l == 0:
                    y_l_node = self.taxonomy.root_node
                else:
                    y_l_node = self.taxonomy.node_at_level_from_node(
                        l, y_node)
                if l + 1 == y_level:
                    y_l_child_node = y_node
                else:
                    y_l_child_node = self.taxonomy.node_at_level_from_node(
                        l + 1, y_node)

                # Update the denominator count
                y_l_node.data['prob_correct_counts'][1] += 1.

                if l < z_level:
                    # Get the child at `l+1` for the worker's prediction
                    if l + 1 == z_level:
                        z_l_child_node = z_node
                    else:
                        z_l_child_node = self.taxonomy.node_at_level_from_node(
                            l + 1, z_node)
                    # Are the children nodes the same?
                    if y_l_child_node == z_l_child_node:
                        # Update the numerator count for the parent
                        y_l_node.data['prob_correct_counts'][0] += 1.

        # compute the skills
        for node in self.taxonomy.breadth_first_traversal():
            if not node.is_leaf:

                pooled_node = self.params.taxonomy.nodes[node.key]

                # Prob correct
                num = self.params.prob_correct_beta * pooled_node.data['prob_correct'] + node.data['prob_correct_counts'][0]
                denom = self.params.prob_correct_beta + node.data['prob_correct_counts'][1]
                denom = np.clip(denom, a_min=0.00000001, a_max=None)
                node.data['prob_correct'] = np.clip(num / denom, a_min=0.00000001, a_max=0.99999)

        # Placeholder for skills
        total_num_correct = 0.
        for image in self.images.itervalues():
            y = image.y.label
            z = image.z[self.id].label
            if y == z:
                total_num_correct += 1.

        prob = total_num_correct / max(0.0001, len(self.images))
        self.prob = prob
        self.skill = [self.prob]

        # Estimate our probability of trusting previous annotations by looking
        # at our agreement with previous annotations
        if self.params.model_worker_trust:
            prob_trust_num = self.params.prob_trust_beta * self.params.prob_trust
            prob_trust_denom = self.params.prob_trust_beta

            for image in self.images.itervalues():

                if self.params.recursive_trust:
                    # We are only dependent on the annotation immediately
                    # before us.
                    our_t = image.z.keys().index(self.id)
                    if our_t > 0:
                        our_label = image.z[self.id].label
                        prev_anno = image.z.values()[our_t - 1]

                        prob_trust_denom += 1.
                        if our_label == prev_anno.label:
                            prob_trust_num += 1.
                else:
                    # We treat each previous label independently
                    our_label = image.z[self.id].label
                    for prev_worker_id, prev_anno in image.z.iteritems():
                        if prev_worker_id == self.id:
                            break
                        if not prev_anno.is_computer_vision() or self.params.naive_computer_vision:
                            prob_trust_denom += 1.
                            if our_label == prev_anno.label:
                                prob_trust_num += 1.

            self.prob_trust = np.clip(
                prob_trust_num / float(prob_trust_denom), 0.00000001, 0.9999)
            self.skill.append(self.prob_trust)

    def parse(self, data):
        super(CrowdWorkerMulticlassSingleBinomial, self).parse(data)
        if 'taxonomy_data' in data:
            self.taxonomy = Taxonomy()
            self.taxonomy.load(data['taxonomy_data'])
            self.taxonomy.finalize()

    def encode(self):
        data = super(CrowdWorkerMulticlassSingleBinomial, self).encode()
        if self.taxonomy is not None:
            data['taxonomy_data'] = self.taxonomy.export()
        return data


class CrowdLabelMulticlassSingleBinomial(CrowdLabel):
    """ A multiclass label.
    """

    def __init__(self, image, worker, label=None):
        super(CrowdLabelMulticlassSingleBinomial, self).__init__(image, worker)

        self.label = label
        self.gtype = 'multiclass_single_bin'

    def compute_log_likelihood(self):
        """ The likelihood of the label.
        """
        y = self.image.y.label
        z = self.label

        taxonomy = self.worker.taxonomy

        y_node = taxonomy.nodes[y]
        y_level = y_node.level

        z_node = taxonomy.nodes[z]
        z_level = z_node.level

        ll = 0.

        # Sum the likelihoods from each layer of the taxonomy for the user
        # providing this annotation
        for l in xrange(0, z_level):

            # Get the child at `l+1` for the worker's prediction
            if l + 1 == z_level:
                z_l_child_node = z_node
            else:
                z_l_child_node = taxonomy.node_at_level_from_node(
                    l + 1, z_node)

            # probability of the user selecting this node (pooled probability)
            z_l_child_node_prob = z_l_child_node.data['prob']

            # If we are deeper than y or at its level, than just sum the
            # probability of selecting the node
            if l >= y_level:
                ll += math.log(z_l_child_node_prob)

            else:
                if l == 0:
                    y_l_node = taxonomy.root_node
                else:
                    y_l_node = taxonomy.node_at_level_from_node(l, y_node)

                # Probability of the worker labeling children of this node
                # correctly
                y_l_node_prob_correct = y_l_node.data['prob_correct']

                if l + 1 == y_level:
                    y_l_child_node = y_node
                else:
                    y_l_child_node = taxonomy.node_at_level_from_node(
                        l + 1, y_node)

                # The worker was "correct"
                if y_l_child_node == z_l_child_node:
                    ll += math.log(y_l_node_prob_correct)

                else:
                    ll += (math.log(1. - y_l_node_prob_correct) +
                           math.log(z_l_child_node_prob))

        if self.worker.params.model_worker_trust:
            # Agreement with the previous worker annotations
            if self.worker.params.recursive_trust:
                # Recursive computation
                t = self.image.z.keys().index(self.worker.id)
                p = self.worker.compute_prob_of_previous_annotations(self.image.id, z, t, compute_denom=False)
                ll += math.log(p)
            else:
                # Assume previous labels are treated independently
                prob_z = self.image.params.class_probs[z]
                our_worker_id = self.worker.id
                for worker_id, prev_anno in self.image.z.iteritems():
                    if worker_id == our_worker_id:
                        break
                    if not prev_anno.is_computer_vision() or self.image.params.naive_computer_vision:
                        if z == prev_anno.label:
                            ll += math.log(self.worker.prob_trust)
                        else:
                            ll += (math.log(1. - self.worker.prob_trust) + math.log(prob_z))

        return ll

    def loss(self, y):
        return 1. - float(self.label == y.label)

    def parse(self, data):
        super(CrowdLabelMulticlassSingleBinomial, self).parse(data)
        self.label = self.label
