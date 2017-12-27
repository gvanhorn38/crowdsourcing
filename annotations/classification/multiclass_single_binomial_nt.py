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

Multiclass annotations, but no taxonomy.

Simple attempt to make the common case fast.

Assume labels are integers in the range [0, # classes)

"""

from collections import Counter
import copy
import math
import random

import numpy as np

from ...crowdsourcing import CrowdDataset, CrowdImage, CrowdWorker, CrowdLabel


class CrowdDatasetMulticlass(CrowdDataset):

    def __init__(self,

                 class_probs=None,  # numpy array

                 # Global priors for the probability of a class occuring, used to estimate the class priors
                 class_probs_prior_beta=10,
                 class_probs_prior=None, # numpy array

                 # Global priors used to compute the pooled probabilities
                 prob_correct_prior_beta=15,
                 prob_correct_prior=0.8,
                 # Pooled probability across all workers
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
                 recursive_trust=False,

                 **kwargs):

        super(CrowdDatasetMulticlass, self).__init__(**kwargs)
        self._CrowdImageClass_ = CrowdImageMulticlass
        self._CrowdWorkerClass_ = CrowdWorkerMulticlass
        self._CrowdLabelClass_ = CrowdLabelMulticlass

        # Class probabilities
        self.class_probs = class_probs
        self.class_probs_prior_beta = class_probs_prior_beta
        self.class_probs_prior = class_probs_prior

        # Worker correctness probabilities
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

    def copy_parameters_from(self, dataset, full=True):
        super(CrowdDatasetMulticlass, self).copy_parameters_from(
            dataset, full=full)
        self.class_probs = copy.copy(dataset.class_probs)
        self.class_probs_prior_beta = dataset.class_probs_prior_beta
        self.class_probs_prior = copy.copy(dataset.class_probs_prior)

        self.prob_correct_prior_beta = dataset.prob_correct_prior_beta
        self.prob_correct_prior = dataset.prob_correct_prior
        self.prob_correct_beta = dataset.prob_correct_beta
        self.prob_correct = dataset.prob_correct

        self.prob_trust_prior_beta = dataset.prob_trust_prior_beta
        self.prob_trust_prior = dataset.prob_trust_prior
        self.prob_trust_beta = dataset.prob_trust_beta
        self.prob_trust = dataset.prob_trust

        self.model_worker_trust = dataset.model_worker_trust
        self.recursive_trust = dataset.recursive_trust

        self.estimate_priors_automatically = dataset.estimate_priors_automatically

    def estimate_priors(self, gt_dataset=None):
        """Estimate the dataset-wide worker skill priors. This is used to regularize per worker skill parameters.
        We want to estimate the following:
          The probability for each class: p(y)
          The probability that any given worker is correct: p(z|y) = p^c

        These probabilities are all Bernoulli with Beta priors, so the estimate is of the form:
          (m + a) / (m + a + l + b)
          m = sum of positive events
          a = a parameter from the Beta distribution
          l = sum of negative events
          b = b parameter from the Beta distribution
        """

        # Probabiilty of a worker being correct (with a Beta prior)
        prob_correct_num = self.prob_correct_prior_beta * self.prob_correct_prior
        prob_correct_denom = self.prob_correct_prior_beta
        class_counts = {y: 0. for y in self.class_probs}

        for i in self.images:

            # Does this image have a computer vision annotation?
            has_cv = 0
            if self.cv_worker and self.cv_worker.id in self.images[i].z:
                has_cv = 1

            # Skip this image if it doesn't have any annotations
            # NOTE: if an image has only one annotion, do we want to estimate parameters from it?
            if len(self.images[i].z) - has_cv <= 1:
                continue

            # If we have access to a ground truth dataset, then use the label from there.
            if not gt_dataset is None:
                y = gt_dataset.images[i].y.label
            # Otherwise, grab the current prediction for the image
            else:
                y = self.images[i].y.label

            class_counts[y] += 1.

            # Go through each worker and add their annotation to the respective present / not present counts.
            for w in self.images[i].z:
                # Skip the computer vision annotations
                if not self.images[i].z[w].is_computer_vision():
                    z = self.images[i].z[w].label

                    if z == y:
                        prob_correct_num += 1.

                    # We need to "double count" here. Otherwise the numerator will be much bigger than the denominator.
                    prob_correct_denom += 1.

        self.prob_correct = np.clip(
            prob_correct_num / float(prob_correct_denom), 0.00000001, 0.9999)

        # Probability of a given class (with a Beta prior)
        self.class_probs = {}
        num_images = float(np.sum(class_counts.values()))
        for y in self.class_probs_prior:
            num = self.class_probs_prior[y] * \
                self.class_probs_prior_beta + class_counts[y]
            denom = self.class_probs_prior_beta + num_images
            self.class_probs[y] = np.clip(
                num / denom, a_min=0.00000001, a_max=0.99999)

        # Probability of a worker trusting previous annotations (with a Beta prior)
        prob_trust_num = self.prob_trust_prior_beta * self.prob_trust_prior
        prob_trust_denom = self.prob_trust_prior_beta

        for worker_id, worker in self.workers.iteritems():
            for image in worker.images.itervalues():

                worker_t = image.z.keys().index(worker_id)
                if worker_t > 0:
                    worker_label = image.z[worker_id].label
                    prev_anno = image.z.values()[worker_t - 1]

                    prob_trust_denom += 1.
                    if worker_label == prev_anno.label:
                        prob_trust_num += 1.

        self.prob_trust = np.clip(
            prob_trust_num / float(prob_trust_denom), 0.00000001, 0.9999)

    def initialize_parameters(self, avoid_if_finished=False):
        """Pass on the dataset-wide worker skill priors to the workers.
        """
        for worker in self.workers.itervalues():
            if avoid_if_finished and worker.finished:
                continue
            worker.prob_correct = self.prob_correct
            worker.prob_trust = self.prob_trust

    def parse(self, data):
        super(CrowdDatasetMulticlass, self).parse(data)
        if 'class_probs' in data:
            self.class_probs = np.array(data['class_probs'])
        if 'class_probs_prior' in data:
            self.class_probs_prior = np.array(data['class_probs_prior'])

    def encode(self):
        data = super(CrowdDatasetMulticlass, self).encode()
        data['class_probs'] = self.class_probs.tolist()
        data['class_probs_prior'] = self.class_probs_prior.tolist()
        return data

class CrowdImageMulticlass(CrowdImage):
    def __init__(self, id_, params):
        super(CrowdImageMulticlass, self).__init__(id_, params)

        self.risk = 1.

    def crowdsource_simple(self, avoid_if_finished=False):
        """Simply do majority vote.
        """
        if avoid_if_finished and self.finished:
            return

        votes = Counter([anno.label for anno in self.z.values()]).items()
        if len(votes) > 0:
            votes.sort(key=lambda x: x[1])
            votes.reverse()
            max_votes = votes[0][1]
            contenders = [x[0] for x in votes if x[1] == max_votes]

            pred_y = random.choice(contenders)
        else:
            # BUG: could bias towards class priors
            pred_y = random.choice(range(len(self.params.class_probs)))
        self.y = CrowdLabelMulticlass(image=self, worker=None, label=pred_y)

    def predict_true_labels(self, avoid_if_finished=False):
        """ Compute the y that is most likely given the annotations, worker skills, etc.
        """

        if avoid_if_finished and self.finished:
            return

        class_probs = self.params.class_probs
        log_class_probs = np.log(class_probs)
        num_classes = class_probs.shape[0]
        class_labels = np.arange(num_classes)

        ncv = self.params.naive_computer_vision

        num_workers = sum([1 for anno in self.z.itervalues() if not anno.is_computer_vision() or ncv])

        if self.params.model_worker_trust and num_workers > 1:

            # Pull out the relevant data from the workers
            worker_labels = []
            worker_prob_trust = []
            worker_prob_correct = []
            for anno in self.z.itervalues():
                if not anno.is_computer_vision() or ncv:
                    worker_labels.append(anno.label)
                    worker_prob_trust.append(anno.worker.prob_trust)
                    worker_prob_correct.append(anno.worker.prob_correct)

            num_workers = len(worker_labels)
            worker_labels = np.array(worker_labels)
            worker_prob_trust = np.array(worker_prob_trust)
            worker_prob_correct = np.array(worker_prob_correct)

            ###############################
            # # Version 1:

            # # Probability of the annotations given a specific true class and worker skill
            # # p(z_j | y_i, w_j)
            # # For a single binomial model, this takes one of two values:
            # # w_j
            # # (1 - w_j) p(z)
            # prob_anno = np.empty((num_workers, num_classes), dtype=np.float)
            # for wind in xrange(num_workers):
            #     wl = worker_labels[wind]
            #     pc = worker_prob_correct[wind]
            #     pnc = 1. - pc
            #     ppnc = pnc * class_probs[wl]
            #     prob_anno[wind] = np.where(class_labels == wl, pc, ppnc)

            # # p(H^{t-1} | z_j, w_j)
            # # Each worker j will represent a row. Each column z will represent
            # # a class. Each entry [j, z] will be the probability of the previous
            # # annotations given that the worker j provided label z.
            # prob_prior_responses = np.empty((num_workers, num_classes), dtype=np.float)

            # # Fill in the first two rows as the base case.
            # prob_prior_responses[0,:] = 1. #class_probs

            # pl = worker_labels[0]
            # wl = worker_labels[1]
            # pt = worker_prob_trust[1]
            # pnt = 1. - pt
            # #ppnt = (1. - pt) * class_probs[wl]
            # prob_prior_responses[1,:] = np.where(class_labels == pl, pt, pnt)

            # # Fill in the subsequent rows for each additional worker
            # for wind in xrange(2, num_workers):
            #     # We want to compute the probability of the previous annotation
            #     # for each possible answer that this worker could have given

            #     pl = worker_labels[wind-1]
            #     wl = worker_labels[wind]
            #     pt = worker_prob_trust[wind]
            #     pnt = 1. - pt
            #     #ppnt = pnt * class_probs[wl]

            #     perception = np.where(class_labels == pl, pt, pnt)
            #     num = perception * prob_prior_responses[wind - 1]
            #     denom = num.sum()
            #     prob_prior_responses[wind] = num / denom

            # # Store these computions with the labels, to be used when computing the log likelihood.
            # # wind = 0
            # # for anno in self.z.values():
            # #     if not anno.is_computer_vision() or ncv:
            # #         wl = worker_labels[wind]
            # #         anno.prob_prev_annos = prob_prior_responses[wind, wl]
            # #         wind += 1

            # # Compute the probability of the annotations given specific
            # # ground truth classes
            # # p(z_j | y, H, w)
            # num = prob_anno * prob_prior_responses
            # denom = num.sum(axis = 1)
            # probs = num / denom[:, np.newaxis]

            # # Compute log(p(y)) + Sum( log(p(z_j | y, H, w)) )
            # lprobs = np.log(probs).sum(axis = 0)
            # lls_v1 = log_class_probs + lprobs
            #################################

            #################################
            # Version 2:
            # placeholder matrix for future computations
            #temp_mat = np.empty((num_classes, num_classes), dtype=np.float)
            #temp_ind_mat = np.full((num_classes, num_classes), fill_value=False, dtype=np.bool)
            #np.fill_diagonal(temp_ind_mat, True)
            temp_ind_mat = np.eye(num_classes, dtype=np.bool)
            #temp_inds = np.array([False] * num_classes, dtype=np.bool)
            #temp_inds[0] = True

            # p(H^{t-1} | z_j, w_j)
            # Each worker j will represent a row. Each column z will represent
            # a class. Each entry [j, z] will be the probability of the previous
            # annotations given that the worker j provided label z.
            prob_prior_responses = np.empty((num_workers, num_classes), dtype=np.float)

            # Fill in the first two rows as the base case.
            prob_prior_responses[0,:] = 1.
            pl = worker_labels[0]
            wl = worker_labels[1]
            pt = worker_prob_trust[1]
            pnt = 1. - pt
            #ppnt = (1. - pt) * class_probs[wl]
            prob_prior_responses[1,:] = np.where(class_labels == pl, pt, pnt)

            # Fill in the subsequent rows for each additional worker
            #t_denom = np.empty(num_classes)
            for wind in xrange(2, num_workers):
                # We want to compute the probability of the previous annotation
                # for each possible answer that this worker could have given

                pl = worker_labels[wind-1]
                wl = worker_labels[wind]
                pt = worker_prob_trust[wind]
                pnt = 1. - pt
                ppnt = pnt * class_probs[wl]

                # p(z_j^{t-1} | z, w_j^t) * p(H^{t-2} | z_j^{t-1}, w_j^{t-1})
                num = np.where(class_labels == pl, pt, ppnt) * prob_prior_responses[wind - 1][pl]

                # Sum( p(z | z_j^t, w_j^t) * p(H^{t-2} | z, w_j^{t-1}) )
                #a = np.full(shape=(num_classes, num_classes), fill_value=pnt, dtype=np.float)
                # temp_mat.fill(pnt)
                # np.fill_diagonal(temp_mat, pt)
                # b = temp_mat * prob_prior_responses[wind - 1]
                # denom = np.sum(b, axis=1)

                diag = pt * prob_prior_responses[wind - 1]
                r = ppnt * prob_prior_responses[wind - 1]
                #denom_iter = (np.where(np.roll(temp_inds, c), diag, r).sum() for c in xrange(num_classes))
                #denom = np.fromiter(denom_iter, dtype=np.float, count=num_classes)
                #denom = np.array([np.where(class_labels == c, diag, r).sum() for c in xrange(num_classes)])
                #denom = np.array([np.where(np.roll(temp_inds, c), diag, r).sum() for c in xrange(num_classes)])
                denom = np.where(temp_ind_mat, diag, r).sum(axis=1)

                #for c in xrange(num_classes):
                #    t_denom[c] = np.where(class_labels == c, diag, r).sum()

                # if np.allclose(denom, t_denom):
                #     print "g2g"

                prob_prior_responses[wind] = num / denom

            # Store these computions with the labels, to be used when computing the log likelihood.
            wind = 0
            for anno in self.z.values():
                if not anno.is_computer_vision() or ncv:
                    wl = worker_labels[wind]
                    anno.prob_prev_annos = prob_prior_responses[wind, wl]
                    wind += 1

            probs = np.empty((num_workers, num_classes))
            for wind in xrange(num_workers):
                wl = worker_labels[wind]
                pc = worker_prob_correct[wind]
                pnc = 1. - pc
                ppnc = pnc * class_probs[wl]

                num = np.where(class_labels == wl, pc, ppnc) * prob_prior_responses[wind][wl]

                #a = np.full(shape=(num_classes, num_classes), fill_value=ppnc, dtype=np.float)
                # temp_mat.fill(ppnc)
                # np.fill_diagonal(temp_mat, pc)
                # b = temp_mat * prob_prior_responses[wind]
                # denom = np.sum(b, axis=1)

                diag = pc * prob_prior_responses[wind]
                r = ppnc * prob_prior_responses[wind]
                #denom_iter = (np.where(np.roll(temp_inds, c), diag, r).sum() for c in xrange(num_classes))
                #denom = np.fromiter(denom_iter, dtype=np.float, count=num_classes)
                #denom = np.array([np.where(class_labels == c, diag, r).sum() for c in xrange(num_classes)])
                #denom = np.array([np.where(np.roll(temp_inds, c), diag, r).sum() for c in xrange(num_classes)])
                denom = np.where(temp_ind_mat, diag, r).sum(axis=1)

                # if np.allclose(denom, denom_2):
                #     print "g2g"

                probs[wind] = num / denom

            # Compute log(p(y)) + Sum( log(p(z_j | y, H, w)) )
            lprobs = np.log(probs).sum(axis = 0)
            lls = log_class_probs + lprobs

            # if np.allclose(lls, lls_v1):
            #     print "same thing %d" % (num_workers,)
            # else:
            #     print "NOPE %d" % (num_workers,)

        else:
            # Not modeling worker trust, or there is only 1 annotation on the image

            # Each row of this matrix will store the log probabilities that need to be
            # summed to get the log likelihood of a class. The first column will store
            # the log(p(y)) of the class, the subsequent columns will store (for each
            # worker) either log(w_j) or (log(1-w_j) + log(z))
            ll_to_sum = np.empty((num_classes, len(self.z) + 1), dtype=np.float)
            ll_to_sum[:,0] = log_class_probs

            worker_index = 1 # if there is a computer vision annotation, then we won't fill in all of the columns
            for anno in self.z.itervalues():
                if not anno.is_computer_vision() or ncv:

                    wl = anno.label
                    pc = anno.worker.prob_correct
                    lpc = np.log(pc)
                    lpnc = np.log(1. - pc)
                    lppnc = lpnc + log_class_probs[wl]

                    ll_to_sum[:,worker_index] = np.where(class_labels == wl, lpc, lppnc)
                    worker_index += 1
            lls = np.sum(ll_to_sum[:,:worker_index], axis=1)


        y_labels = np.argsort(lls)[::-1]
        lls = lls[y_labels]

        if self.id == '4515275':
            print y_labels[:10]
            print lls[:10]

            print log_class_probs[y_labels][:2]
            print lprobs[y_labels][:2]

            print
            print np.log(probs).sum(axis=0)[y_labels][:2]
            print np.log(prob_prior_responses).sum(axis=0)[y_labels][:2]
            print

            print
            print np.log(probs)[:,y_labels][:,:2]
            print np.log(prob_prior_responses)[:,y_labels][:,:2]
            print

            print num_workers
            print worker_labels
            print worker_prob_correct
            print worker_prob_trust


        pred_y = y_labels[0]
        self.y = CrowdLabelMulticlass(image=self, worker=None, label=pred_y)

        m = lls[0]
        num = 1.
        denom = np.sum(np.exp(lls - m))
        prob_y = num / denom
        self.risk = 1. - prob_y

    def compute_log_likelihood(self):
        """Compute the log likelihood of the predicted label given the prior that the class is present.
        The probability that we are labeled k is Bernoulli with parameter equal to the probability that
        the class is present.
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


class CrowdWorkerMulticlass(CrowdWorker):
    def __init__(self, id_, params):
        super(CrowdWorkerMulticlass, self).__init__(id_, params)
        self.skill = None

        # Copy over the global probabilities
        self.prob_correct = params.prob_correct
        self.prob_trust = params.prob_trust

    def compute_log_likelihood(self):
        """ The log likelihood of the skill is simply the log of the Beta distribution.
        """
        ll = ((self.params.prob_correct * self.params.prob_correct_beta - 1) * math.log(self.prob_correct) +
              ((1 - self.params.prob_correct) * self.params.prob_correct_beta - 1) * math.log(1. - self.prob_correct))

        if self.params.model_worker_trust:

            ll += ((self.params.prob_trust * self.params.prob_trust_beta - 1) * math.log(self.prob_trust) +
                ((1 - self.params.prob_trust) * self.params.prob_trust_beta - 1) * math.log(1. - self.prob_trust))

        return ll

    def estimate_parameters(self, avoid_if_finished=False):
        """
        """
        # For each worker, we have a binomial distribution for the probability a worker is correct.
        # This distribution has a Beta prior from the distribution of all workers pooled together.

        # Estimate our probability of being correct by looking at our agreement with predicted labels
        num_correct = self.params.prob_correct_beta * self.params.prob_correct
        num_total = self.params.prob_correct_beta
        for image in self.images.itervalues():

            if len(image.z) <= 1:
                continue

            y = image.y.label
            z = image.z[self.id].label
            num_total += 1
            if y == z:
                num_correct += 1

        num_total = float(max(num_total, 0.0000001))  # just in case
        self.prob_correct = np.clip(num_correct / num_total, 0.00001, 0.99999)
        self.skill = [self.prob_correct]

        # Estimate our probability of trusting previous annotations by looking at our agreement with previous
        # annotations
        if self.params.model_worker_trust:
            prob_trust_num = self.params.prob_trust_beta * self.params.prob_trust
            prob_trust_denom = self.params.prob_trust_beta

            for image in self.images.itervalues():

                # We are only dependent on the annotation immediately before us.
                our_t = image.z.keys().index(self.id)
                if our_t > 0:
                    our_label = image.z[self.id].label
                    prev_anno = image.z.values()[our_t - 1]

                    prob_trust_denom += 1.
                    if our_label == prev_anno.label:
                        prob_trust_num += 1.

            self.prob_trust = np.clip(
                prob_trust_num / float(prob_trust_denom), 0.00000001, 0.9999)

            self.skill.append(self.prob_trust)


class CrowdLabelMulticlass(CrowdLabel):
    def __init__(self, image, worker, label=None):
        super(CrowdLabelMulticlass, self).__init__(image, worker)

        self.label = label
        self.gtype = 'multiclass_single_bin'

        # Computed when estimating the image labels
        self.prob_prev_annos = None

    def compute_log_likelihood(self):
        """ The likelihood of the label is simply the Bernoulli.
        """
        y = self.image.y.label
        z = self.label

        # NOTE: do we want to use the cv predictions here?
        # if self.image.cv_pred is not None and not self.image.params.naive_computer_vision:
        #  prob_z = self.image.cv_pred.prob[z]
        # else:
        #  prob_z = self.image.params.class_probs[z]

        # NOTE: should this just be passed in to the constructor?
        prob_z = self.image.params.class_probs[z]

        # Agreement between the anno and the predicted label
        if y == z:
            ll = math.log(self.worker.prob_correct)
        else:
            ll = math.log(1. - self.worker.prob_correct) + math.log(prob_z)

        # Agreement with the previous worker annotations
        if self.worker.params.model_worker_trust:

            # Should have been computed when estimating the labels
            ll += math.log(self.prob_prev_annos)

        return ll

    def loss(self, y):
        return 1. - float(self.label == y.label)

    def parse(self, data):
        super(CrowdLabelMulticlass, self).parse(data)
        self.label = int(self.label)  # Assume all labels are ints
