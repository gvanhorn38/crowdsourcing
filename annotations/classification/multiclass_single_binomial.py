"""
MIT License

Copyright (c) 2018 Grant Van Horn

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

from __future__ import absolute_import, division#, print_function

from collections import Counter
import math
import os
import random

import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag as sparse_block_diag

from ...crowdsourcing import CrowdDataset, CrowdImage, CrowdWorker, CrowdLabel
from ...util.taxonomy import Taxonomy

import ctypes
from numpy.ctypeslib import ndpointer
so_fp = os.path.join(os.path.dirname(__file__), "annoprobs.so")
lib = ctypes.cdll.LoadLibrary(so_fp)
get_class_lls = lib.compute_log_likelihoods
get_class_lls.restype = None
get_class_lls.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ndpointer(ctypes.c_float),
    ndpointer(ctypes.c_float),
    ndpointer(ctypes.c_float),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_int),
    ndpointer(ctypes.c_float)
]

class CrowdDatasetMulticlassSingleBinomial(CrowdDataset):
    """ A dataset for multiclass labeling across a taxonomy.
    """

    def __init__(self,

                 # Taxonomy of label nodes
                 taxonomy=None,

                 # Prior probability on the classes: p(y)
                 # An iterable, {class_key : probability of class}
                 class_probs=None, # This will be computed from the taxonomy.
                 # Global priors for the probability of a class occuring,
                 # used to estimate the class priors
                 class_probs_prior_beta=10,
                 # An iterable, {class_key : prior probability of class}
                 class_probs_prior=None, # This will be computed from the taxonomy.

                 # Global priors used to compute the pooled probabilities for a
                 # worker being correct
                 prob_correct_prior_beta=15,
                 prob_correct_prior=0.8,
                 prob_correct_beta=10,
                 prob_correct=0.8,

                 # Global priors used to compute worker trust
                 prob_trust_prior_beta=15,
                 prob_trust_prior=0.8,
                 prob_trust_beta=10,
                 prob_trust=0.8,

                 # TODO: refactor to `verification_task`
                 model_worker_trust=False,
                 # TODO: refactor to `dependent_verification`
                 recursive_trust=True,
                 **kwargs):

        super(CrowdDatasetMulticlassSingleBinomial, self).__init__(**kwargs)
        self._CrowdImageClass_ = CrowdImageMulticlassSingleBinomial
        self._CrowdWorkerClass_ = CrowdWorkerMulticlassSingleBinomial
        self._CrowdLabelClass_ = CrowdLabelMulticlassSingleBinomial

        #self.class_probs = class_probs
        self.class_probs_prior_beta = class_probs_prior_beta
        #self.class_probs_prior = class_probs_prior

        # These are the dataset wide priors that we use to estimate the per
        # worker skill parameters
        self.taxonomy = taxonomy
        self.prob_correct_prior_beta = prob_correct_prior_beta
        self.prob_correct_prior = prob_correct_prior
        self.prob_correct_beta = prob_correct_beta
        self.prob_correct = prob_correct

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

        raise NotImplemented()

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

    def initialize_data_structures(self):
        """ Build data structures to speed up processing.
        """

        assert self.taxonomy.finalized, "The taxonomy must be finalized."
        assert self.taxonomy.priors_initialized, "The taxonomy priors must be initialized."

        #######################
        # Requirements on the taxonomy:
        # No annotations can occur at the root
        # For a parent node, children are sorted by descendant depth, deepest first.


        ########################
        # Assign integer ids to each node in the taxonomy.
        orig_node_key_to_integer_id = {}
        integer_id_to_orig_node_key = {}
        leaf_node_key_set = set()
        leaf_integer_ids = []
        inner_node_integer_ids = [] # [num_nodes-1]
        # We are going to number the nodes based on breadth first search.
        for integer_id, node in enumerate(self.taxonomy.breadth_first_traversal()):
            orig_node_key_to_integer_id[node.key] = integer_id
            integer_id_to_orig_node_key[integer_id] = node.key
            if node.is_leaf:
                leaf_node_key_set.add(node.key)
                leaf_integer_ids.append(integer_id)
            else:
                if not node.is_root:
                    inner_node_integer_ids.append(integer_id)
        self.orig_node_key_to_integer_id = orig_node_key_to_integer_id
        self.integer_id_to_orig_node_key = integer_id_to_orig_node_key
        self.leaf_integer_ids = np.array(leaf_integer_ids, dtype=np.int32)
        self.leaf_node_key_set = leaf_node_key_set
        self.leaf_node_keys = list(leaf_node_key_set)
        self.inner_node_integer_ids = np.array(inner_node_integer_ids, dtype=np.int32)
        self.encode_exclude['orig_node_key_to_integer_id'] = True
        self.encode_exclude['integer_id_to_orig_node_key'] = True
        self.encode_exclude['leaf_integer_ids'] = True
        self.encode_exclude['leaf_node_key_set'] = True
        self.encode_exclude['leaf_node_keys'] = True
        self.encode_exclude['inner_node_integer_ids'] = True

        # Now its important that the remaining data structures respect the ordering
        # that we just created.
        num_nodes = len(self.taxonomy.nodes)

        # Create a node prior vector
        node_priors = np.zeros(num_nodes, dtype=np.float32)
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            node_priors[integer_id] = node.data['prob']
        self.node_priors = node_priors
        self.encode_exclude['node_priors'] = True

        # Store the paths from the root node to each other node.
        # NOTE: this is redundant with path_to_node below.
        root_to_node_path_list = {}
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]

            # Path from the parent node to root
            orig_path = [ancestor.key for ancestor in node.ancestors]
            # Path from the root to the parent node
            orig_path.reverse()
            # Path from the root to the node
            orig_path += [node.key]
            # Get the integer ids for this path
            integer_id_path = [orig_node_key_to_integer_id[k] for k in orig_path]

            root_to_node_path_list[integer_id] = integer_id_path
        self.root_to_node_path_list = root_to_node_path_list
        self.encode_exclude['root_to_node_path_list'] = True

        # Create a map that goes from an internal node's integer id to it's
        # parent index in the skill vector
        internal_node_integer_id_to_skill_vector_index = {}
        skill_vector_index = 0
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            if not node.is_leaf:
                internal_node_integer_id_to_skill_vector_index[integer_id] = skill_vector_index
                skill_vector_index += 1
        self.internal_node_integer_id_to_skill_vector_index = internal_node_integer_id_to_skill_vector_index
        self.encode_exclude['internal_node_integer_id_to_skill_vector_index'] = True

        # And we want an array of size [num_nodes -1] that we will use to index into the skill vector
        # when computing the N vector
        skill_vector_N_indices = np.zeros([num_nodes -1], dtype=np.int32)
        for integer_id in xrange(1, num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            parent_node = node.parent
            parent_integer_id = orig_node_key_to_integer_id[parent_node.key]
            parent_skill_vector_index = internal_node_integer_id_to_skill_vector_index[parent_integer_id]
            skill_vector_N_indices[integer_id-1] = parent_skill_vector_index
        self.skill_vector_N_indices = skill_vector_N_indices
        self.encode_exclude['skill_vector_N_indices'] = True
        #
        #########################


        #########################
        # The following data structures are used to compute p(z | y, w) using
        # a workers M[num_nodes -1][num_nodes -1] matrix and N[num_nodes -1] vector
        # These computations don't involve the root node, hence the -1 everywhere.

        # For each child node this stores the parent index
        # This is used to construct the N matrix for workers.
        parent_indices = [] # [num_nodes -1]
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            if not node.is_root:
                parent_integer_id = orig_node_key_to_integer_id[node.parent.key]
                parent_indices.append(parent_integer_id)
        self.parent_indices = np.array(parent_indices, np.int32) - 1 # shift everthing down by 1 (accounting for the loss of the root)
        self.encode_exclude['parent_indices'] = True



        # sanity check that the parent indices are increasing
        for i in xrange(1, num_nodes-1):
            assert parent_indices[i-1] <= parent_indices[i]

        # For each child node, store its level
        levels = [] # [num_nodes -1]
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            if not node.is_root:
                levels.append(node.level)
        self.levels = np.array(levels, dtype=np.int32)
        self.encode_exclude['levels'] = True

        max_path_length = self.taxonomy.max_depth + 1

        # For each child node, store the path to the node, padded with -1
        path_to_node = np.zeros([num_nodes-1, max_path_length], dtype=np.int32)
        for integer_id in xrange(1, num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            ancestors = [orig_node_key_to_integer_id[a.key] for a in node.ancestors]
            ancestors.reverse()
            for i, a in enumerate(ancestors):
                path_to_node[integer_id - 1, i] = a
        path_to_node = path_to_node - 1 # shift everthing down by 1 (accounting for the loss of the root)
        self.path_to_node = path_to_node
        self.encode_exclude['path_to_node'] = True

        # Each worker will have a sparse matrix M whose raveled block diagonal length will be scs:
        children_count = [len(n.children) for n in self.taxonomy.inner_nodes()]
        scs = sum([c ** 2 for c in children_count])
        self.scs = scs
        self.encode_exclude['scs'] = True

        # For each node, we want to store how to get to the correct "row" in M (which is a raveled sparse matrix)
        M_offset_indices = [] # [num_nodes - 1]
        # We also want to store how many entries are in that row (i.e. how many siblings the node has)
        num_siblings = [] # [num_nodes - 1]

        block_starts = []

        # We will be building a A = [num_inner_nodes - 1, num_nodes -1] matrix for each worker.
        # For each node we want to the row of its parent in A.
        # We will store -1 for those nodes whose parent is the root node
        node_integer_id_to_A_index = {0 : -1} # should only have num_inner_nodes entries
        parent_offset_when_excluding_leaves = [] # [num_nodes -1]

        current_M_index = 0 # The negative 1 is taken into acount here.
        current_A_index = 0 # The negative 1 is taken into acount here.
        current_block_index = 0
        for integer_id in xrange(1, num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]

            M_offset_indices.append(current_M_index)
            current_M_index += len(node.parent.children)

            num_siblings.append(len(node.parent.children))

            block_starts.append(current_block_index)
            if integer_id < (num_nodes - 1):
                if parent_indices[integer_id - 1] != parent_indices[integer_id]:
                    current_block_index += len(node.parent.children)


            parent_integer_id = orig_node_key_to_integer_id[node.parent.key]
            parent_index_in_A = node_integer_id_to_A_index[parent_integer_id]
            parent_offset_when_excluding_leaves.append(parent_index_in_A)

            if not node.is_leaf:
                node_integer_id_to_A_index[integer_id] = current_A_index
                current_A_index += 1
        self.M_offset_indices = np.array(M_offset_indices, dtype=np.int32)
        self.num_siblings = np.array(num_siblings, dtype=np.int32)
        self.parent_offset_when_excluding_leaves = np.array(parent_offset_when_excluding_leaves, dtype=np.int32)
        self.block_starts = np.array(block_starts, dtype=np.int32)
        self.encode_exclude['M_offset_indices'] = True
        self.encode_exclude['num_siblings'] = True
        self.encode_exclude['parent_offset_when_excluding_leaves'] = True
        self.encode_exclude['block_starts'] = True

        assert max(parent_offset_when_excluding_leaves) == len(inner_node_integer_ids) - 1
        #
        #########################

        #########################
        # Construct index arrays that will be used to create the M matrix for each worker.
        # Since we are using a single binomial model, we can store one value in a vector
        # of length `num_inner_nodes` for each worker.

        # M is going to be a block diagonal matrix
        skill_vector_correct_read_indices = []
        M_correct_indices = []

        skill_vector_incorrect_read_indices = []
        skill_vector_node_priors_read_indices = []
        M_incorrect_indices = []

        cur_block_index = 0
        skill_vector_index = 0
        for integer_id in xrange(num_nodes):
            k = integer_id_to_orig_node_key[integer_id]
            parent_node = self.taxonomy.nodes[k]
            if not parent_node.is_leaf:
                num_children = len(parent_node.children)
                children_integer_ids = [orig_node_key_to_integer_id[k] for k in parent_node.children]

                # Read num_children times from the skill vector at the parent node
                skill_vector_correct_read_indices += [skill_vector_index] * num_children
                # Write to the diagonal
                for c in range(num_children):
                    M_correct_indices.append(cur_block_index + c * num_children + c)


                # Fill in the off diagonal of the block
                # M[r, c] is the probability worker says c when the ground truth is r
                for r in range(num_children):
                    for c, child_integer_id in enumerate(children_integer_ids):
                        if r == c:
                            continue

                        # Read from the skill vector
                        skill_vector_incorrect_read_indices.append(skill_vector_index)
                        # We'll multiply the skill value by the prior of the row node
                        skill_vector_node_priors_read_indices.append(child_integer_id)

                        # Write into M[r,c]
                        M_incorrect_indices.append(cur_block_index + r * num_children + c)

                cur_block_index += num_children ** 2
                skill_vector_index += 1

        # Sanity check, we should be writing to every index in M
        assert len(set(M_incorrect_indices)) == len(M_incorrect_indices)
        assert len(set(M_correct_indices)) == len(M_correct_indices)
        assert len(set(M_incorrect_indices).intersection(M_correct_indices)) == 0
        M_indices = M_incorrect_indices + M_correct_indices
        M_indices.sort()
        assert M_indices == range(scs)

        self.skill_vector_correct_read_indices = np.array(skill_vector_correct_read_indices, np.intp)
        self.M_correct_indices = np.array(M_correct_indices, np.intp)
        self.skill_vector_incorrect_read_indices = np.array(skill_vector_incorrect_read_indices, np.intp)
        self.skill_vector_node_priors_read_indices = np.array(skill_vector_node_priors_read_indices, np.intp)
        self.M_incorrect_indices = np.array(M_incorrect_indices, np.intp)
        self.encode_exclude['skill_vector_correct_read_indices'] = True
        self.encode_exclude['M_correct_indices'] = True
        self.encode_exclude['skill_vector_incorrect_read_indices'] = True
        self.encode_exclude['skill_vector_node_priors_read_indices'] = True
        self.encode_exclude['M_incorrect_indices'] = True

        # Construct a vector holding the default skill priors for a worker
        self.default_skill_vector = np.ones(self.taxonomy.num_inner_nodes, dtype=np.float32) * self.prob_correct
        self.pooled_prob_correct_vector = np.ones(self.taxonomy.num_inner_nodes, dtype=np.float32) * self.prob_correct_prior
        self.encode_exclude['default_skill_vector'] = True
        self.encode_exclude['pooled_prob_correct_vector'] = True


    def estimate_priors(self, gt_dataset=None):
        """Estimate the dataset-wide parameters.
        For the full dataset (given a gt_dataset) we want to estimate the class
        priors.
        """

        # Initialize the `prob_correct_prior`, `prob_correct` and `prob` for each node
        if not self.taxonomy.priors_initialized:
            print("INITIALIZING all node priors to defaults")
            self.initialize_default_priors()

        # Pooled counts
        # For the single binomial taxonomic model, we are learning a single parameter
        # at each internal node of the taxonomy.
        skill_counts_num = np.zeros_like(self.pooled_prob_correct_vector)
        skill_counts_denom = np.zeros_like(self.pooled_prob_correct_vector)

        class_dist = {node.key: 0. for node in self.taxonomy.leaf_nodes()}

        # NOTE: experimental...
        # We can limit the effect of power users by sampling at most X annotations from each worker
        worker_sample_counts = {} # worker id to the number of annotations we have sampled
        max_worker_sample = 20

        for image_id, image in self.images.iteritems():

            has_cv = 0
            if self.cv_worker and self.cv_worker.id in image.z:
                has_cv = 1

            # Skip this image if it doesn't have at least 2 human annotations.
            if len(image.z) - has_cv <= 1:
                continue

            # If we have access to a ground truth dataset, then use the label
            # from there.
            if gt_dataset is not None:
                y = gt_dataset.images[i].y.label
            # Otherwise, grab the current prediction for the image
            else:
                y = image.y.label

            # Update the class distributions
            class_dist[y] += 1.

            y_integer_id = self.orig_node_key_to_integer_id[y]
            y_node_list = self.root_to_node_path_list[y_integer_id]

            # Go through each worker and add their annotation to the respective
            # counts.
            for worker_id in image.z:
                # Skip the computer vision annotations
                if not image.z[worker_id].is_computer_vision():

                    if worker_id not in worker_sample_counts:
                        worker_sample_counts[worker_id] = 0
                    if worker_sample_counts[worker_id] >= max_worker_sample:
                        continue
                    worker_sample_counts[worker_id] += 1

                    z_integer_id = self.orig_node_key_to_integer_id[image.z[worker_id].label]
                    z_node_list = self.root_to_node_path_list[z_integer_id]

                    # Traverse the paths and update the counts.
                    # Note that we update the parent count when the children match
                    for child_node_index in range(1, len(y_node_list)):
                        y_parent_node = y_node_list[child_node_index - 1]
                        # update the denominator
                        skill_vector_index = self.internal_node_integer_id_to_skill_vector_index[y_parent_node]
                        skill_counts_denom[skill_vector_index] += 1

                        if child_node_index < len(z_node_list):
                            y_node = y_node_list[child_node_index]
                            z_node = z_node_list[child_node_index]
                            if y_node == z_node:
                                skill_counts_num[skill_vector_index] += 1

        # NOTE: `self.prob_correct_prior` should probably be a per node value
        num = self.prob_correct_prior_beta * self.prob_correct_prior + skill_counts_num
        denom = self.prob_correct_prior_beta + skill_counts_denom
        denom = np.clip(denom, a_min=0.00000001, a_max=None)
        self.pooled_prob_correct_vector = np.clip(num / denom, a_min=0.00000001, a_max=0.99999)


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

        # Create a node prior vector
        num_nodes = len(self.taxonomy.nodes)
        node_priors = np.zeros(num_nodes, dtype=np.float32)
        for integer_id in xrange(num_nodes):
            k = self.integer_id_to_orig_node_key[integer_id]
            node = self.taxonomy.nodes[k]
            node_priors[integer_id] = node.data['prob']
        self.node_priors = node_priors


        # Probability of a worker trusting previous annotations
        # (with a Beta prior)
        if self.model_worker_trust:
            prob_trust_num = self.prob_trust_prior_beta * self.prob_trust_prior
            prob_trust_denom = self.prob_trust_prior_beta

            # We can limit the effect of power users by sampling at most X annotations from each worker
            worker_sample_counts = {} # worker id to the number of annotations we have sampled
            max_worker_sample = 20

            for worker_id, worker in self.workers.iteritems():
                for image in worker.images.itervalues():

                    if worker_id not in worker_sample_counts:
                        worker_sample_counts[worker_id] = 0
                    if worker_sample_counts[worker_id] >= max_worker_sample:
                        continue
                    worker_sample_counts[worker_id] += 1

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

            worker.skill_vector = np.copy(self.default_skill_vector)

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

        leaf_node_key_set = self.params.leaf_node_key_set
        labels = [anno.label for anno in self.z.values() if anno.label in leaf_node_key_set]
        votes = Counter(labels)
        if len(votes) > 0:
            pred_y = votes.most_common(1)[0][0]

        else:
            # Randomly pick a label.
            # NOTE: could bias with the class priors.
            pred_y = random.choice(self.params.leaf_node_keys)

        self.y = CrowdLabelMulticlassSingleBinomial(
            image=self, worker=None, label=pred_y)

    def _get_worker_labels_and_trust(self):
        # Collect the relevant data from each worker to build the prob_prior_responses tensor.

        ncv = self.params.naive_computer_vision
        num_workers = sum([1 for anno in self.z.itervalues() if not anno.is_computer_vision() or ncv])

        worker_labels = np.empty(num_workers, dtype=np.int32)
        worker_prob_trust = np.empty(num_workers, dtype=np.float32)

        w = 0
        for anno in self.z.itervalues():
            if not anno.is_computer_vision() or ncv:
                integer_label = self.params.orig_node_key_to_integer_id[anno.label]
                worker_labels[w] = integer_label
                worker_prob_trust[w] = anno.worker.prob_trust
                w += 1

        return worker_labels, worker_prob_trust

    def _make_probability_of_prior_responses(self, worker_labels, worker_prob_trust, node_priors):
        """ Build a tensor that stores the probability of prior annotations given that a worker said "z".
        Args:
            worker_labels (numpy.ndarray int32): The worker annotations, should be in the range [0, num_nodes)
            worker_prob_trust (numpy.ndarray float32): The probability of a worker trusting previous annotations.
            node_priors (numpy.ndarray float32): The probability of each node occuring (this should include the root node).
        """

        num_nodes = node_priors.shape[0]
        node_labels = np.arange(num_nodes - 1)
        num_workers = worker_labels.shape[0]

        # Build a tensor that stores the probability of prior annotations given that a worker
        # said "z"
        # p(H^{t-1} | z_j, w_j)
        # Each worker j will represent a row. Each column z will represent
        # a node. Each entry [j, z] will be the probability of the previous
        # annotations given that the worker j provided label z.
        # Use 1s as the default value.
        prob_prior_responses = np.ones((num_workers, num_nodes -1), dtype=np.float32)
        if num_workers > 1:
            previous_label = worker_labels[0] - 1 # to account for the loss of the root node
            worker_label = worker_labels[1] # this indexes into node_prios, so don't subtract 1
            pt = worker_prob_trust[1]
            pnt = 1. - pt
            ppnt = (1. - pt) * node_priors[worker_label]
            # Put pt in the spot where the previous label occurs, put ppnt in all other spots
            prob_prior_responses[1] = np.where(node_labels == previous_label, pt, ppnt)

            # Fill in the subsequent rows for each additional worker
            for wind in xrange(2, num_workers):

                previous_label = worker_labels[wind-1] -1 # minus 1 to account for the loss of the root node
                worker_label = worker_labels[wind]
                pt = worker_prob_trust[wind]
                pnt = 1. - pt
                ppnt = pnt * node_priors[worker_label]

                # For each possible value of z, we want to compute p(H^{t-1} | z, w_j^t)

                # Numerator : p(z_j^{t-1} | z, w_j^t) * p(H^{t-2} | z_j^{t-1}, w_j^{t-1})
                # Let X = p(H^{t-2} | z_j^{t-1}, w_j^{t-1}), which is fixed
                # The numerator values will be ppnt * X, except for the location
                # where z == previous label, which will have the value pt * X
                ppr = prob_prior_responses[wind - 1][previous_label] # p(H^{t-2} | z_j^{t-1}, w_j^{t-1})
                num = np.full(shape=num_nodes-1, fill_value=ppnt * ppr, dtype=np.float32) # fill in (1 - p_j)p(z_j) * p(H^{t-2} | z_j, w_j) for all values
                num[previous_label] = pt * ppr # (this is where z == previous_label)

                # Denominator: Sum( p(z | z_j^t, w_j^t) * p(H^{t-2} | z, w_j^{t-1}) )
                # For each possible value of z, we will have 1 location where z == previous_label
                match = pt * prob_prior_responses[wind - 1]
                # For each possible value of z, we will have (num_nodes-1) locations where z != previous label
                no_match = ppnt * prob_prior_responses[wind - 1]
                no_match_sum = no_match.sum()
                # For each possible value of z, add the single match value with all no match values, and subtract off the no match value that is actually a match
                denom = (match + no_match_sum) - no_match

                # p(H^{t-1} | z_j, w_j)
                prob_prior_responses[wind] = num / denom

        return prob_prior_responses

    def _make_M_and_N_matrices(self, num_workers, num_nodes):

        ncv = self.params.naive_computer_vision

        M = np.empty([num_workers, self.params.scs], dtype=np.float32)
        N = np.empty([num_workers, num_nodes -1], dtype=np.float32)
        w = 0
        for anno in self.z.itervalues():
            if not anno.is_computer_vision() or ncv:
                anno.worker.build_M_and_N(M[w], N[w])
                w += 1
        return M, N

    def _compute_class_log_likelihoods(self, avoid_if_finished=False):

        node_priors = self.params.node_priors
        leaf_node_indices = self.params.leaf_integer_ids
        class_priors = node_priors[leaf_node_indices]

        num_nodes = node_priors.shape[0]
        num_classes = leaf_node_indices.shape[0]

        # Collect the relevant data from each worker to build the prob_prior_responses tensor.
        worker_labels, worker_prob_trust = self._get_worker_labels_and_trust()
        num_workers = worker_labels.shape[0]
        ncv = self.params.naive_computer_vision

        # Build a tensor that stores the probability of prior annotations given that a worker
        # said "z"
        # p(H^{t-1} | z_j, w_j)
        # Each worker j will represent a row. Each column z will represent
        # a node. Each entry [j, z] will be the probability of the previous
        # annotations given that the worker j provided label z.
        prob_prior_responses = self._make_probability_of_prior_responses(worker_labels, worker_prob_trust, node_priors)

        # Store these computions with the labels, to be used when computing the log likelihood.
        wind = 0
        for anno in self.z.values():
            if not anno.is_computer_vision() or ncv:
                wl = worker_labels[wind] -1 # to account for the loss of the root node
                anno.prob_prev_annos = prob_prior_responses[wind, wl]
                wind += 1
        # NOTE: This is tricky. If we are estimating worker trust, then we
        # actually want to compute the log likelihood of the annotations.
        # In which case we will return after doing that.
        if avoid_if_finished and self.finished:
            return

        M, N = self._make_M_and_N_matrices(num_workers, num_nodes)

        w = num_workers
        n = num_nodes - 1
        l = num_classes
        scs = self.params.scs

        M_offset_indices = self.params.M_offset_indices # [n]
        num_siblings = self.params.num_siblings # [n]
        parents = self.params.parent_indices # [n]
        inner_nodes = self.params.inner_node_integer_ids - 1 # shift things down to account for the root.
        leaf_nodes = leaf_node_indices - 1
        parent_offset_when_excluding_leaves = self.params.parent_offset_when_excluding_leaves
        block_starts = self.params.block_starts
        worker_labels = worker_labels - 1

        lls = np.zeros(num_classes, dtype=np.float32)

        # Get the log likelihood of each class
        get_class_lls(
            w, n, l, scs,
            M, N, prob_prior_responses,
            M_offset_indices, num_siblings,
            parents, inner_nodes, leaf_nodes, parent_offset_when_excluding_leaves, block_starts,
            worker_labels,
            lls
        )

        # NOTE: Debugging stuff...
        # if self.id == "1442526":
        #     print np.argmax(lls)
        #     print self.params.leaf_integer_ids[np.argmax(lls)]
        #     print lls[np.argmax(lls)]
        #     print np.log(class_priors)[np.argmax(lls)]
        #     print parents[438]
        #     print parents[2088]
        #     print num_siblings[438]
        #     print num_siblings[2088]
        #     print parent_offset_when_excluding_leaves[438]
        #     print parent_offset_when_excluding_leaves[2088]
        #     print self.params.root_to_node_path_list[439]
        #     print self.params.root_to_node_path_list[2089]
        #     print prob_prior_responses[:,438]
        #     print prob_prior_responses[:,2088]

        # Tack on the class priors
        class_log_likelihoods = lls + np.log(class_priors)

        # if self.id == "1442526":
        #     print np.argmax(class_log_likelihoods)
        #     print self.params.leaf_integer_ids[np.argmax(class_log_likelihoods)]
        #     print lls[np.argmax(class_log_likelihoods)]
        #     print np.log(class_priors)[np.argmax(class_log_likelihoods)]
            # l = worker_labels[0] - 1
            # for wid in range(num_workers):
            #     print M[wid][M_offset_indices[l]:M_offset_indices[l] + num_siblings[l]]

        return class_log_likelihoods

    def predict_true_labels(self, avoid_if_finished=False):
        """ Compute the y that is most likely given the annotations, worker
        skills, etc.
        """

        # NOTE: This is tricky. If we are estimating worker trust, then we
        # actually want to compute the log likelihood of the annotations below.
        # In which case we will return after doing that.
        if avoid_if_finished and self.finished and not self.params.model_worker_trust:
            return None

        class_log_likelihoods = self._compute_class_log_likelihoods(avoid_if_finished)

        if avoid_if_finished and self.finished:
            return class_log_likelihoods

        # Get the most likely prediction
        arg_max_index = np.argmax(class_log_likelihoods)
        pred_y_integer_id = self.params.leaf_integer_ids[arg_max_index]

        pred_y = self.params.integer_id_to_orig_node_key[pred_y_integer_id]
        self.y = CrowdLabelMulticlassSingleBinomial(image=self, worker=None, label=pred_y)

        # Compute the risk of the predicted label
        # Subtract the maximum value for numerical stability
        m = class_log_likelihoods[arg_max_index]
        num = 1.
        denom = np.sum(np.exp(class_log_likelihoods - m))
        prob_y = num / denom
        self.risk = 1. - prob_y

        # NOTE: Debugging stuff...
        # if self.id == "1442526":# '4267920':#'8754692':# '1112246': #
        #     sort_idxs = np.argsort(class_log_likelihoods)[::-1]
        #     y_labels = self.params.leaf_integer_ids[sort_idxs]
        #     print "Predicted Label:"
        #     print pred_y_integer_id
        #     print "Most likely classes:"
        #     print y_labels
        #     print "Log likelihoods:"
        #     print class_log_likelihoods[sort_idxs]
        #     print "Log Class priors"
        #     class_priors = self.params.node_priors[self.params.leaf_integer_ids]
        #     print np.log(class_priors)[sort_idxs]
        #     #print "Log Anno Probs"
        #     #print lls[sort_idxs]
        #     print "Prob y"
        #     print prob_y

        #     print "Worker info:"
        #     print len(self.workers)
        #     worker_labels, worker_prob_trust = self._get_worker_labels_and_trust()
        #     print worker_labels
        #     #print M
        #     print worker_prob_trust
        #     # are the worker labels at leaf nodes or not?
        #     is_leaf_anno = []
        #     tax = self.params.taxonomy
        #     for worker_label in worker_labels:
        #         if tax.nodes[self.params.integer_id_to_orig_node_key[worker_label]].is_leaf:
        #             is_leaf_anno.append(1)
        #         else:
        #             is_leaf_anno.append(0)
        #     print is_leaf_anno
        #     print tax.nodes[self.params.integer_id_to_orig_node_key[worker_label]].level # print the level

        #     print np.log(class_priors)[worker_labels[0]]

        #     print worker_labels[0] in self.params.leaf_integer_ids

        return class_log_likelihoods

    def compute_probability_of_each_leaf_node(self, class_log_likelihoods=None):
        """ Compute the probability of each leaf node being the correct label.
        Returns:
            numpy.ndarray of size [num_leaf_nodes] that represents the probability of each leaf node being the correct label.
        """
        if class_log_likelihoods is None:
            class_log_likelihoods = self._compute_class_log_likelihoods()

        # transform back to probabilities:
        class_exp = np.exp(class_log_likelihoods)
        class_probabilities = class_exp / np.maximum(0.00000001, class_exp.sum())

        return class_probabilities

    def compute_probability_of_each_node(self, class_log_likelihoods=None):
        """ Compute the probability of the leaf nodes, and then roll the probabilities up the taxonomy.
        Returns:
            numpy.ndarray of size [num_nodes] that represents the probability of each node occuring.
        """
        leaf_node_probabilities = self.compute_probability_of_each_leaf_node(class_log_likelihoods)

        node_probs = np.zeros(self.params.node_priors.shape[0], dtype=np.float32)

        for zero_indexed_leaf_integer_id in xrange(leaf_node_probabilities.shape[0]):
            # Map the 0-indexed leaf id to its node integer id
            y = self.params.leaf_integer_ids[zero_indexed_leaf_integer_id]
            path_to_y = self.params.root_to_node_path_list[y]
            # Add the probability to all nodes on the path to y
            node_probs[path_to_y] += leaf_node_probabilities[zero_indexed_leaf_integer_id]

        return node_probs


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

        self.skill_vector = None
        self.M = None
        self.N = None
        self.encode_exclude['M'] = True
        self.encode_exclude['N'] = True


    def compute_log_likelihood(self):
        """ The log likelihood of the skill.
        """
        ll = 0

        pooled_prob_correct = self.params.pooled_prob_correct_vector
        prob_correct = self.skill_vector

        ll = np.sum(
            (pooled_prob_correct * self.params.prob_correct_beta - 1) * np.log(prob_correct) +
            (( 1. - pooled_prob_correct) * self.params.prob_correct_beta - 1) * np.log( 1. - prob_correct)
        )

        if self.params.model_worker_trust:
            ll += ((self.params.prob_trust * self.params.prob_trust_beta - 1) * math.log(self.prob_trust) +
                   ((1 - self.params.prob_trust) * self.params.prob_trust_beta - 1) * math.log(1. - self.prob_trust))

        return ll


    def estimate_parameters(self, avoid_if_finished=False):
        """ Estimate the worker skill parameters.
        """

        #assert self.taxonomy is not None, "Worker %s's taxonomy was not initialized"

        # For the single binomial taxonomic model, we are learning a single parameter
        # at each internal node of the taxonomy.
        skill_counts_num = np.zeros_like(self.params.pooled_prob_correct_vector)
        skill_counts_denom = np.zeros_like(self.params.pooled_prob_correct_vector)

        internal_node_integer_id_to_skill_vector_index = self.params.internal_node_integer_id_to_skill_vector_index

        for image in self.images.itervalues():

            if len(image.z) <= 1:
                continue

            y_integer_id = self.params.orig_node_key_to_integer_id[image.y.label]
            z_integer_id = self.params.orig_node_key_to_integer_id[image.z[self.id].label]

            y_node_list = self.params.root_to_node_path_list[y_integer_id]
            z_node_list = self.params.root_to_node_path_list[z_integer_id]

            # Traverse the paths and update the counts.
            # Note that we update the parent count when the children match
            for child_node_index in range(1, len(y_node_list)):
                y_parent_node = y_node_list[child_node_index - 1]
                # update the denominator
                skill_vector_index = internal_node_integer_id_to_skill_vector_index[y_parent_node]
                skill_counts_denom[skill_vector_index] += 1

                if child_node_index < len(z_node_list):
                    y_node = y_node_list[child_node_index]
                    z_node = z_node_list[child_node_index]
                    if y_node == z_node:
                        skill_counts_num[skill_vector_index] += 1

        num = self.params.prob_correct_beta * self.params.pooled_prob_correct_vector + skill_counts_num
        denom = self.params.prob_correct_beta + skill_counts_denom
        denom = np.clip(denom, a_min=0.00000001, a_max=None)
        self.skill_vector = np.clip(num / denom, a_min=0.00000001, a_max=0.99999)

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


    def build_M_and_N(self, M, N):
        """
        M is a vector of length `sum_children_squared`
        N is a vector of length `num_nodes -1`
        """

        M[self.params.M_correct_indices] = self.skill_vector[self.params.skill_vector_correct_read_indices]

        # Fill in the off diagonals entries of the block diagonals
        pnc = 1 - self.skill_vector[self.params.skill_vector_incorrect_read_indices]
        ppnc = pnc * self.params.node_priors[self.params.skill_vector_node_priors_read_indices]
        M[self.params.M_incorrect_indices] = ppnc

        # Get the probability of not correct
        pc = self.skill_vector[self.params.skill_vector_N_indices]
        pnc = 1 - pc

        # Multiply the probability of not correct by the prior on the node
        N[:] = self.params.node_priors[1:] * pnc

    # def build_M_and_N_old(self, M, N):
    #     # NOTE: this return M.T!

    #     # Build the M matrix.
    #     # The size will be [num nodes, num nodes]
    #     # M[y, z] is the probability of predicting class z when the true class is y.
    #     #num_nodes = self.params.node_priors.shape[0]
    #     #M = np.zeros([num_nodes - 1, num_nodes -1], dtype=np.float32)

    #     # Fill in the diagaonal of the block diagonals
    #     M[self.params.M_correct_rows, self.params.M_correct_cols] = self.skill_vector[self.params.skill_vector_correct_read_indices]

    #     # Fill in the off diagonals entries of the block diagonals
    #     pnc = 1 - self.skill_vector[self.params.skill_vector_incorrect_read_indices]
    #     ppnc = pnc * self.params.node_priors[self.params.skill_vector_node_priors_read_indices]
    #     M[self.params.M_incorrect_rows, self.params.M_incorrect_cols] = ppnc

    #     # Build the M matrix.
    #     # The size will be [num nodes, num nodes]
    #     # M[y, z] is the probability of predicting class z when the true class is y.
    #     # num_nodes = self.params.node_priors.shape[0]
    #     # M = np.zeros([num_nodes, num_nodes], dtype=np.float32)
    #     # for node_indices in self.params.parent_and_siblings_indices:
    #     #     parent_index = node_indices[0]
    #     #     sibling_indices = node_indices[1:]
    #     #     num_siblings = len(sibling_indices)

    #     #     # Probability of correct and not correct
    #     #     pc = self.skill_vector[parent_index]
    #     #     pnc = 1 - pc

    #     #     # Multiply the probability of not correct by the prior on the node
    #     #     ppnc = self.params.node_priors[sibling_indices] * pnc

    #     #     # M[y,z] = the probability of predicting class z when the true class is y.
    #     #     # Set the off diagonal entries to the probability of not being correct times the node prior
    #     #     M[np.ix_(sibling_indices, sibling_indices)] = np.tile(ppnc, [num_siblings, 1])
    #     #     # Set the diagonal to the probability of being correct
    #     #     M[sibling_indices, sibling_indices] = pc
    #     #self.M = M

    #     # blks = [] # Store a skill value of 1 for the root node (we do this to keep the indexing constant)
    #     # for node_indices in self.params.parent_and_siblings_indices:
    #     #     parent_index = node_indices[0]
    #     #     sibling_indices = node_indices[1:]
    #     #     num_siblings = len(sibling_indices)

    #     #     # Probability of correct and not correct
    #     #     pc = self.skill_vector[parent_index]
    #     #     pnc = 1 - pc

    #     #     # Multiply the probability of not correct by the prior on the node
    #     #     ppnc = self.params.node_priors[sibling_indices] * pnc

    #     #     # Geneate the block matrix, first by tiling the probability of not correct
    #     #     blk = np.tile(ppnc, [num_siblings, 1])
    #     #     # then fill in the diagonal with the probability of being correct
    #     #     np.fill_diagonal(blk, pc)
    #     #     blks.append(blk)
    #     # self.M = block_diag(*blks)
    #     # self.M = sparse_block_diag(blks)

    #     # Build the N vector. This is the probability of a worker selecting a node regardless
    #     # of the true value (i.e. when the parent nodes are different)
    #     #num_nodes = len(self.params.taxonomy.nodes)
    #     #N = np.ones([num_nodes-1], dtype=np.float32) # This will store a value of 1 for the root node. We do this to keep the indexing constant

    #     # Get the probability of not correct
    #     pc = self.skill_vector[self.params.skill_vector_N_indices]
    #     pnc = 1 - pc

    #     # Multiply the probability of not correct by the prior on the node
    #     N[:] = self.params.node_priors[1:] * pnc
    #     #N[1:] = ppnc # don't overwrite the root node.

    #     #return M, N.astype(np.float32)

    def parse(self, data):
        super(CrowdWorkerMulticlassSingleBinomial, self).parse(data)
        if 'skill_vector' in data:
            self.skill_vector = np.array(data['skill_vector'])
            #self.build_M_and_N()
        #if 'taxonomy_data' in data:
        #    self.taxonomy = Taxonomy()
        #    self.taxonomy.load(data['taxonomy_data'])
        #    self.taxonomy.finalize()

    def encode(self):
        data = super(CrowdWorkerMulticlassSingleBinomial, self).encode()
        if self.skill_vector is not None:
            data['skill_vector'] = self.skill_vector.tolist()
        #if self.taxonomy is not None:
        #    data['taxonomy_data'] = self.taxonomy.export()
        return data


class CrowdLabelMulticlassSingleBinomial(CrowdLabel):
    """ A multiclass label.
    """

    def __init__(self, image, worker, label=None):
        super(CrowdLabelMulticlassSingleBinomial, self).__init__(image, worker)

        self.label = label
        self.gtype = 'multiclass_single_bin'
        self.prob_prev_annos = None
        self.encode_exclude['prob_prev_annos'] = True

    def compute_log_likelihood(self):
        """ The likelihood of the label.
        """
        prob_correct = self.worker.skill_vector
        node_probs = self.worker.params.node_priors

        internal_node_integer_id_to_skill_vector_index = self.worker.params.internal_node_integer_id_to_skill_vector_index

        y = self.image.y.label
        z = self.label

        y_integer_id = self.worker.params.orig_node_key_to_integer_id[y]
        z_integer_id = self.worker.params.orig_node_key_to_integer_id[z]

        y_node_list = self.worker.params.root_to_node_path_list[y_integer_id]
        z_node_list = self.worker.params.root_to_node_path_list[z_integer_id]

        ll = 0.
        if len(z_node_list) > 1:

            # pad the y_node_list
            pad_size = max(0, len(z_node_list) - len(y_node_list))
            y_node_list = y_node_list + [-1] * pad_size

            for i in range(len(z_node_list) - 1):
                # Same parents
                if y_node_list[i] == z_node_list[i]:
                    skill_vector_index = internal_node_integer_id_to_skill_vector_index[y_node_list[i]]
                    # Same node
                    if y_node_list[i+1] == z_node_list[i+1]:
                        ll += math.log(prob_correct[skill_vector_index])
                    # Different node
                    else:
                        ll += math.log(1 - prob_correct[skill_vector_index]) + math.log(node_probs[z_node_list[i+1]])
                # Different parents
                else:
                    skill_vector_index = internal_node_integer_id_to_skill_vector_index[z_node_list[i]]
                    ll += math.log(1 - prob_correct[skill_vector_index]) + math.log(node_probs[z_node_list[i+1]])

        if self.worker.params.model_worker_trust:
            # Should have been computed when estimating the labels
            assert self.prob_prev_annos is not None #self.prob_prev_annos = 0.5
            ll += math.log(self.prob_prev_annos)

        return ll

    def loss(self, y):
        return 1. - float(self.label == y.label)

    def parse(self, data):
        super(CrowdLabelMulticlassSingleBinomial, self).parse(data)
        self.label = self.label
