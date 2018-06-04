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

"""
Binary Classification Crowdsourcing.
"""

import math
import random

import numpy as np

from ...crowdsourcing import CrowdDataset, CrowdImage, CrowdWorker, CrowdLabel


# Crowdsourcing for binary classification.  Incorporates a worker skill model
class CrowdDatasetBinaryClassification(CrowdDataset):
    def __init__(self,

                 # Used to estimate dataset wide priors
                 prior_beta=15,
                 prob_present_prior=0.5,
                 prob_present_given_present_prior=0.8,
                 prob_not_present_given_not_present_prior=0.8,

                 # Used to estimate worker skills
                 prob_present_beta=2.5,
                 prob_present=0.5,
                 prob_present_given_present_beta=10,
                 prob_present_given_present=0.8,
                 prob_not_present_given_not_present_beta=10,
                 prob_not_present_given_not_present=0.8,

                 **kwds):
        """
        Args:
          prior_beta: Used in estimating the dataset wide priors: scale multiplier
            for all beta distributions.
          prob_present_prior: Used in estimating the dataset wide priors:
            prior probability of occurence.
          prob_present_given_present_prior: Used in estimating the dataset wide
            priors: probability of a worker saying the object is present when it is
            present.
          prob_not_present_given_not_present_prior: Used in estimating the dataset
            wide priors: probability of a worker saying the object is not present
            when it is not present.
          prob_present_beta: Used in estimating worker skills: scale multiplier for
            the object presence prior beta distribution
          prob_present: Used in estimating the worker skills: the probability that
            the object is present in an image
          prob_present_given_present_beta: Used in estimating worker skills: scale
            multiplier for the beta distribution representing the probability that
            a worker will say the object is present when it is present.
          prob_present_given_present: Used in estimating worker skills: the prior
            probability that a worker will say the object is present when it is
            present.
          prob_not_present_given_not_present_beta: Used in estimating worker
            skills: scale multiplier for the beta distribution representing the
            probability that a worker will say the object is not present when it
            is not present.
          prob_not_present_given_not_present: Used in estimating worker skills:
            the prior probability that a worker will say the object is not present
            when it is not present.
          **kwargs: remaining args that will be passed to CrowdDataset
        """

        super(CrowdDatasetBinaryClassification, self).__init__(**kwds)
        self._CrowdImageClass_ = CrowdImageBinaryClassification
        self._CrowdWorkerClass_ = CrowdWorkerBinaryClassification
        self._CrowdLabelClass_ = CrowdLabelBinaryClassification

        # Variables used to estimate the dataset wide priors
        # When estimating the dataset wide priors, we will overwrite:
        #   self.prob_present
        #   self.prob_present_given_present
        #   self.prob_not_present_given_not_present
        self.prior_beta = prior_beta
        self.prob_present_prior = prob_present_prior
        self.prob_present_given_present_prior = prob_present_given_present_prior
        self.prob_not_present_given_not_present_prior = prob_not_present_given_not_present_prior

        # Variables used to estimate the worker skills
        self.prob_present_beta = prob_present_beta
        self.prob_present = prob_present
        self.prob_present_given_present_beta = prob_present_given_present_beta
        self.prob_present_given_present = prob_present_given_present
        self.prob_not_present_given_not_present_beta = prob_not_present_given_not_present_beta
        self.prob_not_present_given_not_present = prob_not_present_given_not_present

        # Plot data
        self.skill_names = ['Prob Correct Given Present', 'Prob Correct Given Not Present']

        # MTurk data
        name = self.name if self.name and len(self.name) > 0 else "object"
        self.hit_params = {'object_name': name}
        dollars_per_hour, sec_per_click, sec_per_hour = 8, 1.2, 3600
        self.reward = 0.15
        self.images_per_hit = int(math.ceil(self.reward / dollars_per_hour * sec_per_hour / (sec_per_click * self.prob_present_prior)))
        self.description = self.title = "Click on images where " + ('an ' if name[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a ') + name + " is present"
        self.keywords = "images,labelling,present," + name
        self.html_template_dir = 'html/binary'

    def copy_parameters_from(self, dataset, full=True):
        super(CrowdDatasetBinaryClassification, self).copy_parameters_from(dataset, full=full)
        if full:
            self.prob_not_present_given_not_present = dataset.prob_not_present_given_not_present
            self.prob_present_given_present = dataset.prob_present_given_present
            self.prob_present = dataset.prob_present
            self.estimate_priors_automatically = False

    def estimate_priors(self, gt_dataset=None):
        """Estimate the dataset-wide worker skill priors. This is used to regularize per worker skill parameters.
        We want to estimate the following:
          The probability the class is present: p(y=1)
          The probability any worker says present given that the class is present: p(z | y=1) = p^1
          The probability any worker says not present given that the class is not present: p(z| y=0) = p^0

        These probabilities are all Bernoulli with Beta priors, so the estimate is of the form:
          (m + a) / (m + a + l + b)
          m = sum of positive events
          a = a parameter from the Beta distribution
          l = sum of negative events
          b = b parameter from the Beta distribution
        """

        # Add the Beta prior to the initial counts
        num_present = self.prior_beta * self.prob_present_prior
        num_not_present = self.prior_beta * (1. - self.prob_present_prior)
        num_not_present_given_not_present = self.prior_beta * self.prob_not_present_given_not_present_prior
        num_present_given_not_present = self.prior_beta * (1. - self.prob_not_present_given_not_present_prior)
        num_present_given_present = self.prior_beta * self.prob_present_given_present_prior
        num_not_present_given_present = self.prior_beta * (1. - self.prob_present_given_present_prior)

        self.initialize_parameters(avoid_if_finished=True)

        # Now go through and add the actual present / not present counts from the data.
        for i in self.images:

            # Does this image have a computer vision annotation?
            has_cv = 0
            if self.cv_worker and self.cv_worker.id in self.images[i].z:
                has_cv = 1

            # Skip this image if it doesn't have any annotations
            if len(self.images[i].z) - has_cv < 1:
                continue

            # If we have access to a ground truth dataset, then use the label from there.
            if not gt_dataset is None:
                y = gt_dataset.images[i].y.label
            # Otherwise, grab the current prediction for the image
            else:
                y = self.images[i].y.label
                # We prefer the soft label, if available
                if hasattr(self.images[i].y, "soft_label"):
                    y = self.images[i].y.soft_label

            # Go through each worker and add their annotation to the respective present / not present counts.
            for w in self.images[i].z:
                # Skip the computer vision annotations
                if not self.images[i].z[w].is_computer_vision():
                    z = self.images[i].z[w].label

                    # We need to "double count" here. Otherwise the numerator will be much bigger than the denominator.
                    num_present += y
                    num_not_present += 1 - y

                    # We could assume that z is 0 or 1, but we might have a soft z in the future.
                    # Hence we don't use an if else statement here, but rather z and (1. - z).
                    num_not_present_given_not_present += (1. - y) * (1. - z)
                    num_present_given_not_present += (1. - y) * z
                    num_present_given_present += y * z
                    num_not_present_given_present += y * (1. - z)

        # Probability that the class is present.
        num = float(num_present)
        denom = max(0.0001, num_present + num_not_present)
        self.prob_present = np.clip(num / denom, 0.0001, 0.9999)

        # Probability that any worker says not present when the class is not present.
        num = float(num_not_present_given_not_present)
        denom = max(0.0001, num_not_present_given_not_present + num_present_given_not_present)
        self.prob_not_present_given_not_present = np.clip(num / denom, 0.0001, 0.9999)

        # Probabiilty that any worker says present when the class is present.
        num = float(num_present_given_present)
        denom = max(0.0001, num_present_given_present + num_not_present_given_present)
        self.prob_present_given_present = np.clip(num / denom, 0.0001, 0.9999)

    def initialize_parameters(self, avoid_if_finished=False):
        """Pass on the dataset-wide worker skill priors to the workers.

        """

        for w in self.workers:
            self.workers[w].prob_not_present_given_not_present = self.prob_not_present_given_not_present
            self.workers[w].prob_present_given_present = self.prob_present_given_present
            self.workers[w].prob_present = self.prob_present


class CrowdImageBinaryClassification(CrowdImage):
    def __init__(self, id, params):
        super(CrowdImageBinaryClassification, self).__init__(id, params)
        self.original_name = id

        self.cv_pred = None  # This is a CrowdLabel from a computer vision system

    def crowdsource_simple(self, avoid_if_finished=False):
        """Simply do majority vote.

        """
        if avoid_if_finished and self.finished:
            return

        accum = 0
        num_annotations = 0

        for w in self.z:
            if not self.z[w].is_computer_vision():
                accum += self.z[w].label
                num_annotations += 1

        thresh = num_annotations / 2.

        if accum > thresh:
            label = 1.
        elif accum == thresh:
            label = float(random.random() > .5)  # randomly assign 0 or 1
        else:
            label = 0.

        self.y = CrowdLabelBinaryClassification(image=self, worker=None, label=label)

        if accum != thresh:
            self.y.soft_label = label
        else:
            self.y.soft_label = 0.5

    def predict_true_labels(self, avoid_if_finished=False):
        """Compute the log likelihood for the class being present and not present, and predict the more likely one.

        Essentially doing the argmax p(y) * p(z | y, w) for y either 0 or 1

        The probability of a label p(y) can be estimated across the whole dataset.

        The probability of a workers label given y and the skill parameters is Bernoulli.
        """
        if avoid_if_finished and self.finished:
            return

        # Compute the likelihood of p(y)
        if self.cv_pred is not None and not self.params.naive_computer_vision:
            ll_not_present = math.log(1 - self.cv_pred.prob),
            ll_present = math.log(self.cv_pred.prob)
        else:
            ll_not_present = math.log(1 - self.params.prob_present)
            ll_present = math.log(self.params.prob_present)

        # compute the likelihood of (Z | y, W)
        for w in self.z:
            if not self.z[w].is_computer_vision() or self.params.naive_computer_vision:

                if self.z[w].label == 1:
                    ll_present += math.log(self.z[w].worker.prob_present_given_present)
                    ll_not_present += math.log(1 - self.z[w].worker.prob_not_present_given_not_present)
                else:
                    ll_present += math.log(1 - self.z[w].worker.prob_present_given_present)
                    ll_not_present += math.log(self.z[w].worker.prob_not_present_given_not_present)

        # Choose present or not present
        if ll_present > ll_not_present:
            label = 1.
        elif ll_present == ll_not_present:
            label = float(random.random() > .5)  # randomly choose 0 or 1
        else:
            label = 0.
        self.y = CrowdLabelBinaryClassification(image=self, worker=None, label=label)

        self.ll_present = ll_present
        self.ll_not_present = ll_not_present

        # Now we can compute the posterior probability p(y|Z) and the posterior risk l(y,y')p(y|Z) of our crowd label y'
        # We'll work in the mindset of computing p(y = 1 | Z)
        # In this mindset, if our crowd label is 1, then the prob of that label is p(y = 1 | Z) and the risk is 1 - p(y = 1 | Z)
        # If our crowd label is 0, then the prob of that label is 1 - p(y = 1 | Z) and the risk is p(y = 1 | Z)
        # So we can store p(y = 1 | Z) as our probabiity and we can use our crowd label along with the prob to compute the risk
        # risk = prob * (1 - y') + (1 - prob) * y'

        # NOTE: subtracting the max is for numerical stability
        # We are esentially multiplying by e^-m / e^-m
        m = max(ll_present, ll_not_present)
        self.prob = math.exp(ll_present - m) / (math.exp(ll_not_present - m) + math.exp(ll_present - m))
        self.risk = self.prob * (1 - self.y.label) + (1 - self.prob) * self.y.label
        self.y.soft_label = self.prob

    def compute_log_likelihood(self):
        """Compute the log likelihood of the predicted label given the prior that the class is present.
        The probability that we are labeled with a 1 or 0 is Bernoulli with parameter equal to the probability that
        the class is present.
        """
        # Grab our current crowd label
        y = self.y.label
        if hasattr(self.y, "soft_label"):
            y = self.y.soft_label

        if self.cv_pred != None:
            ll = (1 - y) * math.log(1 - self.cv_pred.prob) + y * math.log(self.cv_pred.prob)

        else:
            # NOTE: is it `1 - self.params.prob_correct` for both cases?
            ll = (1 - y) * math.log(1 - self.params.prob_present) + y * math.log(1 - self.params.prob_present)

        return ll

    # Estimate difficulty parameters
    def estimate_parameters(self, avoid_if_finished=False):
        """We didn't bother with the image difficulty parameters for this task.

        """
        if (avoid_if_finished and self.finished) or len(self.z) <= 1:
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


class CrowdWorkerBinaryClassification(CrowdWorker):
    def __init__(self, id, params):
        super(CrowdWorkerBinaryClassification, self).__init__(id, params)
        self.skill = None

        # Copy over the global probabilities
        self.prob_not_present_given_not_present = params.prob_not_present_given_not_present
        self.prob_present_given_present = params.prob_present_given_present
        self.prob_present = params.prob_present

    def compute_log_likelihood(self):
        """Compute the log likelihood of our skill estimates given the prior on these skills.
        Our skill parameters are Beta distributions.
        """
        ll = ((self.params.prob_present_given_present * self.params.prob_present_given_present_beta - 1) * math.log(self.prob_present_given_present) +
              ((1 - self.params.prob_present_given_present) * self.params.prob_present_given_present_beta - 1) * math.log(1 - self.prob_present_given_present))
        ll += ((self.params.prob_not_present_given_not_present * self.params.prob_not_present_given_not_present_beta - 1) * math.log(self.prob_not_present_given_not_present) +
               ((1 - self.params.prob_not_present_given_not_present) * self.params.prob_not_present_given_not_present_beta - 1) * math.log(1 - self.prob_not_present_given_not_present))
        return ll

    def estimate_parameters(self, avoid_if_finished=False):
        # For each worker, we have binomial distributions for 1) the probability a worker thinks a
        # class is present if the class is present in the ground truth, 2) the probability a worker thinks a
        # class is present if the class is not present in the ground truth.  Each of these distributions has a
        # Beta prior from the distribution of all workers pooled together

        num_present = self.params.prob_present_given_present_beta
        num_present_given_present = self.params.prob_present_given_present_beta * self.params.prob_present_given_present

        num_not_present = self.params.prob_not_present_given_not_present_beta
        num_not_present_given_not_present = self.params.prob_not_present_given_not_present_beta * self.params.prob_not_present_given_not_present

        num_times_worker_said_present = 0.

        for image_id, image in self.images.items():
            # Get the current predicted label for the image
            y = image.y.label
            if hasattr(image.y, 'soft_label'):
                y = image.y.soft_label

            # Keep track of the number of images whose current label is 1
            num_present += y

            # Keep track of the number of images whose current label is 0
            num_not_present += 1. - y

            # Get our annotation for this image
            z = image.z[self.id].label

            # Keep track of the number of images this worker labeled 1 and the predicted label is 1
            num_present_given_present += y * z

            # Keep track of the number of images this worker labeled 0 and the predicted label is 0
            num_not_present_given_not_present += (1. - y) * (1. - z)

            # Keep track of the number of times this worker said present
            num_times_worker_said_present += z

        # NOTE: Need to justify this
        # if True:
        #   # Custom prior for class presence
        #   beta = min(self.params.prob_present_beta, num_present + num_not_present)
        #   prob_present = num_times_worker_said_present / max(0.0001, len(self.images))
        #   self.prob_present = prob_present
        # else:
        #   # Use the global prior for class presence
        #   beta = self.params.prob_present_beta
        #   prob_present = self.prob_present # assigned by CrowdDatasetBinaryClassification in `initialize_parameters()`
        beta = 0
        prob_present = 0.5

        # Predict our probability of labeling 1 when the gt is 1
        num_present += beta
        num_present_given_present += beta * prob_present
        self.prob_present_given_present = num_present_given_present / float(num_present)

        # Predict our probability of labeling 0 when the gt is 0
        num_not_present += beta
        num_not_present_given_not_present += beta * (1 - prob_present)
        self.prob_not_present_given_not_present = num_not_present_given_not_present / float(num_not_present)

        self.skill = [self.prob_present_given_present, self.prob_not_present_given_not_present]


class CrowdLabelBinaryClassification(CrowdLabel):
    def __init__(self, image, worker, label=None):
        super(CrowdLabelBinaryClassification, self).__init__(image, worker)
        self.label = label
        self.gtype = 'binary'

        self.prob = 0.5  # default probability of the label. Set by the computer vision worker.

    def compute_log_likelihood(self):
        """Compute the log likelihood of the label, given the predicted label and the worker skills
        The label is Bernoulli, with parameters equal to the worker skills.
        """

        # Get the predicted label
        if hasattr(self.image.y, "soft_label"):
            y = self.image.y.soft_label
        else:
            y = self.image.y.label

        z = self.label

        # We assume that y is "soft" and therefore use this formulation
        # If y is "hard" then we could have done an if else statement conditioned on y
        return (math.log(self.worker.prob_present_given_present) * y * z + math.log(self.worker.prob_not_present_given_not_present) * (1 - y) * (1 - z) +
                math.log(1 - self.worker.prob_present_given_present) * y * (1 - z) + math.log(1 - self.worker.prob_not_present_given_not_present) * (1 - y) * z)

    def loss(self, y):
        return abs(self.label - y.label)

    def parse(self, data):
        super(CrowdLabelBinaryClassification, self).parse(data)
        self.label = float(self.label)
