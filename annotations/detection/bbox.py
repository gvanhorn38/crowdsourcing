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
Bounding box annotations.
"""

import math
import os
import pickle

import numpy as np

from ...crowdsourcing import CrowdDataset, CrowdImage, CrowdWorker, CrowdLabel
from ...util.facility_location import FacilityLocation


# Visualization parameters
MIN_WIDTH = 30
COLORS = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#FFFFFF', '#FFBF4A', '#000080', '#626200', '#006262', '#620062', '#44200F', '#008000']
NUM_COLS = 3

TRUE_POSITIVE_COLOR = '#00FF00'
FALSE_POSITIVE_COLOR = '#FF0000'
GT_COLOR = '#00FF00'
FINISHED_COLOR = '#0000FF'
UNFINISHED_COLOR = '#FFFFFF'
FILL_ALPHA = 0.2

class CrowdDatasetBBox(CrowdDataset):
    """Crowdsourcing images with multiple bounding box labels per image, such that a correspondence
    between annotated boxes of different workers is unknown.  Incorporates a worker skill and image difficulty model
    """

    def __init__(self, **kwds):
        super(CrowdDatasetBBox, self).__init__(**kwds)
        self._CrowdImageClass_ = CrowdImageBBox
        self._CrowdWorkerClass_ = CrowdWorkerBBox
        self._CrowdLabelClass_ = CrowdLabelBBox

        self.prior_sigma_v0 = 5
        self.prob_fp_beta = 5
        self.prob_fn_beta = 5
        self.prior_sigma_image_v0 = 1
        self.prior_sigma_prior = 0.25
        self.prob_fp_prior = .1
        self.prob_fn_prior = .1
        self.prior_sigma = self.prior_sigma_prior
        self.prob_fp = self.prob_fp_prior
        self.prob_fn = self.prob_fn_prior
        self.big_bbox_set = None
        self.encode_exclude['big_bbox_set'] = True
        self.skill_names = ['Prob False Positive', 'Boundary Sigma', 'Prob False Negative']

        # MTurk Data
        name = self.name if self.name and len(self.name) > 0 else "objects"
        self.hit_params = {'object_name': name}
        dollars_per_hour, objects_per_image, sec_per_object, sec_per_hour = 8, 2.5, 3.0, 3600
        self.reward = 0.15
        self.images_per_hit = int(math.ceil(self.reward / dollars_per_hour * sec_per_hour / sec_per_object / objects_per_image))
        self.description = self.title = "Draw boxes around " + name + " in images"
        self.keywords = "boxes,images,labelling," + name
        self.html_template_dir = 'html/bbox'

    def parse(self, data):
        super(CrowdDatasetBBox, self).parse(data)

    def set_true_labels(self, images):
        """ Copy the labels from `images` to `self.images`
        """
        for image_id, image in self.images.iteritems():
            if images[image_id].y is not None:
                if image.y is None:
                    image.y = CrowdLabelBBox(image, None)
                images[image_id].y.copy_into(image.y)
                for worker_id, anno in image.z.iteritems():
                    anno.match_to(image.y)

    def estimate_priors(self, thresh=.5, use_gt=False):
        """ Estimate priors globally over the whole dataset
        """

        # Match worker annotations to true labels
        for i in self.images:
            for w in self.images[i].z:
                y = self.images[i].y if hasattr(self.images[i], 'y') else None
                if not y is None:
                    self.images[i].z[w].match_to(y)

        # Use empirical counts of the number of false positives / false negatives
        # to compute prob_fp and prob_fn
        num_fp, num_p, num_fn, num_tp = self.prob_fp_prior * self.prob_fp_beta, self.prob_fp_beta, self.prob_fn_prior * self.prob_fn_beta, self.prob_fn_beta
        for i in self.images:
            y = self.images[i].y if hasattr(self.images[i], 'y') else None
            has_cv = (1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0)
            if not y is None and not y.bboxes is None and len(self.images[i].z) - has_cv > 1:
                for w in self.images[i].z:
                    z = self.images[i].z[w]
                    if not z.is_computer_vision():
                        matches = [0] * len(y.bboxes)
                        for b in z.bboxes:
                            if b.a is None or b.dist(y.bboxes[b.a]) > thresh:
                                num_fp += 1
                            else:
                                matches[b.a] += 1
                            num_p += 1
                        for j in range(len(y.bboxes)):
                            if matches[j] == 0:
                                num_fn += 1
                        num_tp += len(y.bboxes)
        self.prob_fp = float(num_fp) / num_p
        self.prob_fn = float(num_fn) / num_tp

        # Compute sample variance of annotated bounding boxes compared to matched true bounding boxes (plus scaled-inv-chi priors)
        # and use that to compute worker sigma
        S = self.prior_sigma_v0 * (self.prior_sigma_prior**2)
        num = self.prior_sigma_v0
        for i in self.images:
            y = self.images[i].y if hasattr(self.images[i], 'y') else None
            has_cv = (1 if (self.cv_worker and self.cv_worker.id in self.images[i].z) else 0)
            if not y is None and not y.bboxes is None and len(self.images[i].z) - has_cv > 1:
                for w in self.images[i].z:
                    z = self.images[i].z[w]
                    if not z.is_computer_vision():
                        for b in z.bboxes:
                            if not b.a is None:
                                S += b.dist2(y.bboxes[b.a])
                                num += 1
        self.prior_sigma = math.sqrt(S / num)
        self.initialize_parameters()

    def initialize_parameters(self, avoid_if_finished=False):
        # Initialize worker responses probabilities to the global priors
        for worker_id, worker in self.workers.iteritems():
            worker.prob_fp = self.prob_fp
            worker.prob_fn = self.prob_fn
            worker.sigma = self.prior_sigma

        # Initialize image difficulties to the global priors
        for image_id, image in self.images.iteritems():
            if avoid_if_finished and image.finished:
                continue
            if image.y is not None and image.y.bboxes is not None:
                image.sigmas = [self.prior_sigma for b in image.y.bboxes]

        # Initialize image specific parameters
        for image_id, image in self.images.iteritems():
            if avoid_if_finished and image.finished:
                continue
            if image.y is not None and image.y.bboxes is not None:
                for worker_id, anno in image.z.iteritems():
                    anno.prob_fn = self.prob_fn
                    for b in anno.bboxes:
                        b.sigma = self.prior_sigma
                        if not anno.is_computer_vision() or self.naive_computer_vision:
                            b.prob_fp = self.prob_fp

    def NewCrowdLabel(self, i, w):
        return CrowdLabelBBox(self.images[i], self.workers[w])

    def get_big_bbox_set(self):
        if self.big_bbox_set is None:
            big_bbox_filepath = self.fname + ".big_bbox_set.pkl"
            if os.path.isfile(big_bbox_filepath):
                with open(big_bbox_filepath, "rb") as f:
                    self.big_bbox_set = pickle.load(f)
            else:
                self.build_big_bbox_set()
                with open(big_bbox_filepath, "wb") as f:
                    pickle.dump(self.big_bbox_set, f)
        return self.big_bbox_set

    def build_big_bbox_set(self):
        """ Treat all bounding boxes as if they came from the same image. Create a list of bounding boxes
        whose overlap is less than 0.5 from each other. Compute the probability that these boxes are false
        positives when comparing them with the remaining bounding boxes (that are not included in the list).

        Essentially computes a prior over where objects will be in an image.
        """
        # create a random permutation of the image ids
        images = [i for i in self.images]
        images = [images[i] for i in np.random.permutation(range(len(images)))]

        boxes = []
        num_total = 0
        num = 0
        for image_id in images:
            image = self.images[image_id]
            num += 1
            if image.z is not None:
                for worker_id, anno in image.z.iteritems():
                    if not anno.is_computer_vision():
                        num_total += 1
                        for b in anno.bboxes:
                            bn = SingleBBox(x=b.x / float(anno.image_width), y=b.y / float(anno.image_height),
                                            x2=b.x2 / float(anno.image_width), y2=b.y2 / float(anno.image_height))
                            bn.num = 1
                            good = True
                            for bn2 in boxes:
                                if bn.dist(bn2) < .5:
                                    bn2.num += 1
                                    good = False
                            if good:
                                boxes.append(bn)  # This box did not overlap with any previous box by more than 0.5
        for b in boxes:
            b.prob_fp = b.num / float(num_total)
        self.big_bbox_set = boxes

    def copy_parameters_from(self, dataset, full=True):
        super(CrowdDatasetBBox, self).copy_parameters_from(dataset, full=full)
        if hasattr(dataset, 'big_bbox_set'):
            self.big_bbox_set = dataset.big_bbox_set


def log_gaussian(x, sigma):
    return -.5 * (x / sigma)**2 - .5 * math.log(2 * math.pi) - math.log(sigma)


def dummy_match_cost(b, thresh):
    #    [b is a false positive]  [ maximal gaussian cost]
    return -math.log(max(1e-8, b.prob_fp))  # - log_gaussian(thresh, b.sigma)


def match_cost(b, b_gt, z, d2=None, gt_fixed=False):
    if d2 is None:
        d2 = b.dist2(b_gt)

    # [-Gaussian with x=IOU distance-]   [b is a true pos]
    c = -(log_gaussian(d2, b.sigma) + math.log(max(1e-8, 1 - b.prob_fp)))
    if not gt_fixed:  # gt_fixed signifies that the ground truth is already set (when it isn't set, some terms were stored in openCost)
        #   [Undo false neg in openCost] [ b is a true pos]
        c -= -math.log(max(1e-8, z.prob_fn)) + math.log(max(1e-8, 1 - z.prob_fn))
    return c


class CrowdImageBBox(CrowdImage):
    def __init__(self, id, params):
        super(CrowdImageBBox, self).__init__(id, params)
        self.risk, self.e_fn, self.e_fp, self.e_boundaries_off = None, None, None, None
        self.num_possible_bbox_locs = 10.0
        self.encode_exclude['fc_cache'] = True

    def crowdsource_simple(self, avoid_if_finished=False):
        self.predict_true_labels(avoid_if_finished=avoid_if_finished, simple=True, refine=False)

    def predict_true_labels(self, avoid_if_finished=False, simple=False, refine=False, thresh=.5):
        """ Find a near optimal set of bboxes to be added to the predicted label and which worker boxes
        to assign to each of them, where optimality is defined in terms of the log-likelihood of the joint
        distribution between all worker boxes and predicted boxes. If simple=True, instead, add bboxes to
        the true label if at least 50% of the worker boxes are at least 50% overlapping
        """

        if avoid_if_finished and self.finished:
            return

        big_bbox_set = self.params.get_big_bbox_set()

        worker_ids = [w for w in self.z]

        # Adding a box to the predicted label set without connecting anything to it yet incurs a
        # false negatives for all workers
        openCost = 0
        for k in range(len(worker_ids)):
            zk = self.z[worker_ids[k]]
            if simple:
                openCost += .5  # simple case: open if at least 50% of workers can assign a bbox to it
            else:
                openCost += -math.log(zk.prob_fn)

        dummy = -1  # Connecting a worker box to the dummy facility means it will be left as a false positive
        objs = {}  # Will hold a list of all worker bounding boxes, each represented as a 2-tuple (worker_ind,box_ind)
        openCosts = {}  # Cost of opening up a facility (adding a bounding box to the predicted label)
        objs[dummy] = (-1, -1)
        openCosts[dummy] = 0
        objs_inv = []
        curr_obj = 0
        costs = []

        cityDisallowedCityNeighbors = {}
        for j in range(len(worker_ids)):
            objs_inv.append([])
            for bj_i in range(len(self.z[worker_ids[j]].bboxes)):
                objs_inv[j].append(curr_obj)
                objs[curr_obj] = (j, bj_i)
                curr_obj += 1
            for bj_i in range(len(self.z[worker_ids[j]].bboxes)):
                cityDisallowedCityNeighbors[objs_inv[j][bj_i]] = {}
                for bj_j in range(len(self.z[worker_ids[j]].bboxes)):
                    if bj_i != bj_j:  # a worker's bbox cannot be matched to another bbox he labeled in the same image
                        cityDisallowedCityNeighbors[objs_inv[j][bj_i]][objs_inv[j][bj_j]] = dummy

        for j in range(len(worker_ids)):
            zj = self.z[worker_ids[j]]
            for bj_i in range(len(zj.bboxes)):
                bj = zj.bboxes[bj_i]
                oj = objs_inv[j][bj_i]

                # Matching to the dummy facility means this worker box doesn't get assigned to any predicted box (incurs false positive)
                if simple:
                    costs.append((1, dummy, oj))
                else:
                    costs.append((dummy_match_cost(bj, thresh), dummy, oj))

                # Gaussian cost of matching an object to itself
                if simple:
                    costs.append((0, oj, oj))
                else:
                    costs.append((match_cost(bj, bj, zj, d2=0), oj, oj))

                # Adding a box to the predicted label set without connecting anything to it yet incurs a false negatives for all workers
                openCosts[oj] = openCost

                for k in range(j + 1, len(worker_ids)):
                    zk = self.z[worker_ids[k]]
                    for bk_i in range(len(zk.bboxes)):
                        bk = zk.bboxes[bk_i]
                        ok = objs_inv[k][bk_i]
                        if simple:
                            d = bj.dist(bk)
                            if d < thresh:  # simple case: boxes match if they are at least 50% overlapping
                                costs.append((0, oj, ok))
                                costs.append((0, ok, oj))
                        else:
                            d2 = bj.dist2(bk)
                            if d2 < 1:
                                # matching box zk to zj incurs a gaussian cost, but means that bk won't be a false positive and bj won't be a false negative
                                #            [ ------------------ Gaussian with x=IOU distance ------------------]   [- bk is a true pos -]
                                costs.append((match_cost(bk, bj, zk, d2=d2), oj, ok))

                                # matching box zj to zk incurs a gaussian cost, but means that bj won't be a false positive and bk won't be a false negative
                                #            [ ------------------ Gaussian with x=IOU distance ------------------]   [- bj is a true pos -]
                                costs.append((match_cost(bj, bk, zj, d2=d2), ok, oj))

        # Compute a near optimal set of boxes to add to the predicted label y, and which worker boxes to assign to them
        # print str(self.id)
        [facilities, total_cost] = FacilityLocation(cityFacilityCosts=costs, cityDisallowedCityNeighbors=cityDisallowedCityNeighbors).solve(openFacilityCosts=openCosts, debug=0)
        # print str(self.id) + ": costs=" + str(costs) + " facilities=" + str(facilities) + " openCosts=" + str(openCosts) + " cityDisallowedCityNeighbors=" + str(cityDisallowedCityNeighbors)

        # if not simple and len(self.z)>0 and len(costs)>0:
        #   print "costs=" + str(costs)
        #   print "openCosts=" + str(openCosts)
        #   print "facilities=" + str(facilities)
        #   print self.str()
        #   assert False, "oops"

        # Parse the result of the matching problem, and use that to store the predicted combined labels and assignments of worker boxes to predicted boxes
        self.y = CrowdLabelBBox(self, None)
        self.y.bboxes = []
        for f in facilities:
            if f == dummy:
                for c in facilities[f]:  # boxes assigned to the dummy
                    self.z[worker_ids[objs[c][0]]].bboxes[objs[c][1]].a = None
            else:
                if objs[f][0] >= 0:
                    zf = self.z[worker_ids[objs[f][0]]].bboxes[objs[f][1]]  # boxes assigned to zf
                    b = SingleBBox(x=zf.x, y=zf.y, x2=zf.x2, y2=zf.y2)
                else:
                    bb = big_bbox_set[objs[f][1]]
                    b = SingleBBox(x=bb.x * self.image_width, y=bb.y * self.image_height, x2=bb.x2 * self.image_width, y2=bb.y2 * self.image_height)
                self.y.bboxes.append(b)
                b.est_var = self.params.prior_sigma**2 / len(facilities[f])  # variance in estimation of b
                for c in facilities[f]:
                    self.z[worker_ids[objs[c][0]]].bboxes[objs[c][1]].a = len(self.y.bboxes) - 1
        if refine:
            self.refine_true_label()
        self.sigmas = [self.params.prior_sigma for b in self.y.bboxes]
        self.fc_cache = (costs, openCosts, openCost, cityDisallowedCityNeighbors, facilities, dummy)


        # if self.id == '102523':
        # print "costs=" + str(costs)
        # print "openCosts=" + str(openCosts)
        # print "facilities=" + str(facilities)
        # print "worker_ids=" + str(worker_ids)
        # print "objs=" + str(objs)
        # for w in self.z:
        #     print "predict() i=" + str(self.id) + " w=" + str(w) + " " + str([b.a for b in self.z[w].bboxes]) + '\n'
        # print "y=" + str(self.y.encode()) + '\n'


        # if len(self.z) == 1 and simple:
        #   if len(self.z.values()[0].bboxes) == 1:
        #     assert len(self.y.bboxes) == 1
        #     assert np.isclose(self.z.values()[0].bboxes[0].x, self.y.bboxes[0].x)
        #     assert np.isclose(self.z.values()[0].bboxes[0].y, self.y.bboxes[0].y)
        #     assert np.isclose(self.z.values()[0].bboxes[0].x2, self.y.bboxes[0].x2)
        #     assert np.isclose(self.z.values()[0].bboxes[0].y2, self.y.bboxes[0].y2)

        for w in self.z:
            self.z[w].colorize()

    def compute_expected_false_negatives(self, thresh=.5):
        """ Compute the expected number of false negatives. We will do this by rerunning the facility
        location problem on an instance where:
          1. all cities correspond to bounding boxes that were connected to the dummy node
          2. all facilities are bounding boxes that we not selected to be facilities when computing the predicted labels
          3. no dummy node is available
        For each facility that is opened up (i.e. potential tp missed (so a fn) when computing the predicted labels) we compute the
        cost of opening that facility vs leaving its cities as false positives.
        """

        big_bbox_set = self.params.get_big_bbox_set()
        costs, openCosts, openCost, cityDisallowedCityNeighbors, facilities, dummy = self.fc_cache[:]

        # Continue the above facility location algorithm if we weren't allowed to match anything
        # to the dummy (no false positive worker annotations). Then for each new facility (opened object),
        # measure it's probability when being forced to be opened relative to leaving it as worker false positives
        costs2 = []
        costs_c = {}
        openCosts2 = {}
        self.e_fn = 0
        if dummy in facilities:
            c_f = facilities[dummy]
            for c in c_f:
                costs_c[c] = {}
            for c in costs:
                if c[2] in c_f:  # costs2 will be a subset of costs that only includes entries with a city that was matched to dummy
                    if c[1] not in facilities:
                        costs2.append(c)
                        openCosts2[c[1]] = openCosts[c[1]]
                    costs_c[c[2]][c[1]] = c[0]
            [facilities2, total_cost] = FacilityLocation(cityFacilityCosts=costs2, cityDisallowedCityNeighbors=cityDisallowedCityNeighbors).solve(openFacilityCosts=openCosts2)
            for f in facilities2:
                p_cost = openCosts[f]
                f_cost = 0
                for c in facilities2[f]:
                    p_cost += costs_c[c][f]
                    f_cost += costs_c[c][dummy]
                m = max(p_cost, f_cost)
                self.e_fn += math.exp(-(p_cost - m)) / (math.exp(-(p_cost - m)) + math.exp(-(f_cost - m)))

        # Consider opening other bounding boxes from a larger set than the ones labeled by workers for this image
        for b in big_bbox_set:
            found = False
            for yb in self.y.bboxes:
                if yb.dist(b) < thresh:
                    found = True
                    break
            if not found:
                # If no bounding box from the predicted set matched this "global box" then add the probability that this
                # global box is a false negative.
                self.e_fn += b.prob_fp * math.exp(-openCost)

        return self.e_fn

    # Predict true labels assuming assignments are already known
    def refine_true_label(self):
        x, y, x2, y2, s = [0] * len(self.y.bboxes), [0] * len(self.y.bboxes), [0] * len(self.y.bboxes), [0] * len(self.y.bboxes), [0] * len(self.y.bboxes)
        for w in self.z:
            for b in self.z[w].bboxes:
                if not b.a is None:
                    x[b.a] += b.x / (b.sigma**2)
                    y[b.a] += b.y / (b.sigma**2)
                    x2[b.a] += b.x2 / (b.sigma**2)
                    y2[b.a] += b.y2 / (b.sigma**2)
                    s[b.a] += 1.0 / (b.sigma**2)
        for i in range(len(self.y.bboxes)):
            b = self.y.bboxes[i]
            b.x, b.y, b.x2, b.y2 = x[i] / s[i], y[i] / s[i], x2[i] / s[i], y2[i] / s[i]

    def compute_log_likelihood(self):
        # Scaled-inv-chi-squared prior for sigma parameters
        ll = 0
        for i in range(len(self.y.bboxes)):
            ll += -self.params.prior_sigma_image_v0 * (self.params.prior_sigma**2) / (2 * (self.sigmas[i]**2)) - (1 + self.params.prior_sigma_image_v0 / 2) * math.log(self.sigmas[i]**2)
        return ll / len(self.y.bboxes) if len(self.y.bboxes) > 0 else -self.params.prior_sigma_image_v0 / 2 - (1 + self.params.prior_sigma_image_v0 / 2) * math.log(self.params.prior_sigma**2)

    # Estimate difficulty parameters
    def estimate_parameters(self, avoid_if_finished=False):
        """ Compute sample variance of annotated bounding boxes compared to matched true
        bounding boxes (plus scaled-inv-chi priors) and use that to compute worker sigma.
        """
        if avoid_if_finished and self.finished:
            return

        S = [self.params.prior_sigma_image_v0 * (self.params.prior_sigma**2) for b in self.y.bboxes]
        num = [self.params.prior_sigma_image_v0 for b in self.y.bboxes]  # [2 + self.params.prior_sigma_image_v0 for b in self.y.bboxes]
        for w in self.z:
            z = self.z[w]
            if not z.is_computer_vision():
                for b in z.bboxes:
                    if not b.a is None:
                        S[b.a] += (1 - b.w) * (b.dist2(self.y.bboxes[b.a]) + self.y.bboxes[b.a].est_var)
                        num[b.a] += (1 - b.w)
        self.sigmas = [math.sqrt(S[i] / num[i]) for i in range(len(self.y.bboxes))]

    def check_finished(self, set_finished=True, thresh=.5, loss_fn=1, loss_fp=1):
        """ The risk is a sum of 3 terms:
          1. The expected number of false positives
          2. The expected number of inaccurate true positives
          3. The expected number of false negatives
        """

        if self.finished:
            return True

        # Expected number of false positives
        self.e_fp = 0
        # Expected number of inaccurate true positives
        self.e_boundaries_off = 0
        # Initialize the risk with the expected number of false negatives
        self.risk = self.compute_expected_false_negatives() * loss_fn  # expected number of false negatives in other locations in the image

        # Probability a box is a false positive
        p_fp = [1] * len(self.y.bboxes)
        # Probability a box is a true positive
        p_tp = [1] * len(self.y.bboxes)
        # Sigmas for the boxes
        s = [0] * len(self.y.bboxes)

        for worker_id, anno in self.z.iteritems():
            matches = [None for i in range(len(self.y.bboxes))]
            for b in anno.bboxes:
                if b.a is not None:
                    assert matches[b.a] is None, 'Multiple boxes matched to the same ground truth, image=' + str(self.id) + ", worker=" + str(w) + ", a=" + str(b.a) + " " + str(self.z[w].encode())
                    matches[b.a] = b
                    s[b.a] += 1.0 / (2 * (b.sigma**2))

            for i in range(len(self.y.bboxes)):
                if matches[i] is None:
                    p_fp[i] *= (1 - anno.prob_fn)
                    p_tp[i] *= anno.prob_fn
                else:
                    p_fp[i] *= matches[i].prob_fp
                    p_tp[i] *= (1 - matches[i].prob_fp)  # (1-self.z[w].prob_fn)*(1-matches[i].prob_fp)

        for i in range(len(self.y.bboxes)):
            # probability this ground truth box is a false positive
            self.y.bboxes[i].e_fp = p_fp[i] / (p_fp[i] + p_tp[i])

            # probability this predicted ground truth box is too far away from the true location
            self.y.bboxes[i].e_boundaries_off = 1 - math.erf(thresh * math.sqrt(s[i]))

            self.risk += self.y.bboxes[i].e_fp * loss_fp + (self.y.bboxes[i].e_boundaries_off * (loss_fn + (1 - self.y.bboxes[i].e_fp) * loss_fp))
            self.e_fp += self.y.bboxes[i].e_fp
            self.e_boundaries_off += self.y.bboxes[i].e_boundaries_off

        # print "e_fp: %0.5f" % (self.e_fp,)
        # print "e_boundaries_off: %0.5f" % (self.e_boundaries_off,)
        # print "e_fn: %0.5f" % (self.e_fn,)
        # print "risk: %0.5f" % (self.risk,)
        # print

        finished = bool(self.risk <= self.params.min_risk)
        if set_finished:
            self.finished = finished
        return finished

    def copy_parameters_from(self, image, full=True):
        super(CrowdImageBBox, self).copy_parameters_from(image, full=full)
        if hasattr(image, 'width'):
            self.width = image.width
        if hasattr(image, 'height'):
            self.height = image.height


class CrowdWorkerBBox(CrowdWorker):
    def __init__(self, id, params):
        super(CrowdWorkerBBox, self).__init__(id, params)

        # worker skill parameters
        self.sigma = None  # worker's standard deviation for annotating boundaries of a bbox
        self.prob_fp = None  # probability that a box annotated by this worker will be a false positive
        self.prob_fn = None  # probability that a ground truth box will be missed by this worker
        self.finished = False

    def compute_log_likelihood(self):
        # Beta prior for worker false positive parameter
        ll = ((self.params.prob_fp * self.params.prob_fp_beta - 1) * math.log(self.prob_fp) +
              ((1 - self.params.prob_fp) * self.params.prob_fp_beta - 1) * math.log(1 - self.prob_fp))

        # Beta prior for worker false negative parameter
        ll += ((self.params.prob_fn * self.params.prob_fn_beta - 1) * math.log(self.prob_fn) +
               ((1 - self.params.prob_fn) * self.params.prob_fn_beta - 1) * math.log(1 - self.prob_fn))

        # Scaled-inv-chi-squared prior for sigma parameter
        ll += -self.params.prior_sigma_v0 * (self.params.prior_sigma**2) / (2 * (self.sigma**2)) - (1 + self.params.prior_sigma_v0 / 2) * math.log(self.sigma**2)

        return ll

    def estimate_parameters(self, avoid_if_finished=False):

        if avoid_if_finished and self.finished:
            return

        # Use empirical counts of the number of false positives / false negatives (plus the beta priors)
        # to compute prob_fp and prob_fn
        num_fp = self.params.prob_fp * self.params.prob_fp_beta
        num_p = self.params.prob_fp_beta
        num_fn = self.params.prob_fn * self.params.prob_fn_beta
        num_tp = self.params.prob_fn_beta
        for i in self.images:
            w = (len(self.images[i].z) - 1) / float(len(self.images[i].z))
            z = self.images[i].z[self.id]
            matches = [0] * len(self.images[i].y.bboxes)
            for b in z.bboxes:
                if b.a is None:
                    num_fp += w
                else:  # elif b.dist(self.images[i].y.bboxes[b.a])<.5:
                    matches[b.a] += 1
                num_p += w
            for j in range(len(self.images[i].y.bboxes)):
                if matches[j] == 0:
                    num_fn += w
            num_tp += w * len(self.images[i].y.bboxes)
        self.prob_fp = float(num_fp) / num_p  # probability a given worker box is a false positive
        self.prob_fn = float(num_fn) / num_tp  # probability a worker will miss a particular box in the ground truth label

        # Compute sample variance of annotated bounding boxes compared to matched true bounding boxes (plus scaled-inv-chi priors)
        # and use that to compute worker sigma
        S = self.params.prior_sigma_v0 * (self.params.prior_sigma**2)
        num = self.params.prior_sigma_v0  # + 2
        for i in self.images:
            z = self.images[i].z[self.id]
            for b in z.bboxes:
                if not b.a is None:
                    S += (b.dist2(self.images[i].y.bboxes[b.a]) + self.images[i].y.bboxes[b.a].est_var)  # *b.w
                    num += 1  # b.w
        self.sigma = math.sqrt(S / num)
        self.skill = [self.prob_fp, self.sigma, self.prob_fn]


class CrowdLabelBBox(CrowdLabel):
    def __init__(self, image, worker):
        super(CrowdLabelBBox, self).__init__(image, worker)
        self.bboxes = None  # Array of SingleBBox'es
        self.encode_exclude['bboxes'] = True
        self.gtype = 'bboxes'

    def compute_log_likelihood(self):
        y = self.image.y
        ll = 0
        matches = [0] * len(y.bboxes)
        for b in self.bboxes:
            if b.a is not None:
                matches[b.a] += 1
                ll += -.5 * b.dist2(y.bboxes[b.a]) / (b.sigma**2) - .5 * math.log(2 * math.pi) - math.log(b.sigma) + math.log(1 - self.worker.prob_fp)  # Gaussian
            else:
                ll += math.log(self.worker.prob_fp) - math.log(self.image.num_possible_bbox_locs)
        for i in range(len(y.bboxes)):
            ll += math.log(self.worker.prob_fn) if matches[i] == 0 else math.log(1 - self.worker.prob_fn)
        return ll

    def match_to(self, y, match_by='distance', thresh=.5):
        """
        """
        costs = []
        cityDisallowedCityNeighbors = {}
        for j in range(len(self.bboxes)):
            cityDisallowedCityNeighbors[j] = {}
            for k in range(len(self.bboxes)):
                if j != k:
                    cityDisallowedCityNeighbors[j][k] = -1

        if match_by == 'distance':
            for i in range(len(y.bboxes)):
                for j in range(len(self.bboxes)):
                    d2 = y.bboxes[i].dist2(self.bboxes[j])
                    if d2 < 1:
                        costs.append((d2, i, j))
            for j in range(len(self.bboxes)):
                costs.append((1, -1, j))  # dummy node
        elif match_by == 'prob':
            for i in range(len(y.bboxes)):
                for j in range(len(self.bboxes)):
                    d2 = y.bboxes[i].dist2(self.bboxes[j])
                    # NOTE: when will this not be the case
                    if d2 < 1:
                        costs.append((match_cost(self.bboxes[j], y.bboxes[i], self, d2=d2, gt_fixed=True), i, j))
            for j in range(len(self.bboxes)):
                costs.append((dummy_match_cost(self.bboxes[j], thresh), -1, j))
        else:
            assert False, "match_to(..., match_by=" + match_by + ") not supported"

        [facilities, total_cost] = FacilityLocation(cityFacilityCosts=costs, cityDisallowedCityNeighbors=cityDisallowedCityNeighbors).solve(openFacilityCost=0)
        for f in facilities:
            for c in facilities[f]:
                self.bboxes[c].a = f if f >= 0 else None

        # print "match_to() i=" + str(self.image.id) + " w=" + str(self.worker.id if self.worker else 'null') + " " + str([b.a for b in self.bboxes])
        self.colorize()

    def colorize(self):
        matches = {}
        for b in self.bboxes:
            if not b.a is None:
                if b.a in matches:
                    print 'COLORIZE: Multiple boxes matched to the same ground truth, image=' + str(self.image.id) + ", worker=" + str(self.worker.id if self.worker else 'null') + ", a=" + str(b.a) + " " + str(self.encode())
                    b.a = None
                else:
                    matches[b.a] = b

        for b in self.bboxes:
            if not self.worker is None:
                b.outline_color = FALSE_POSITIVE_COLOR if b.a is None else TRUE_POSITIVE_COLOR
                b.fill_color = None
                b.alpha = 1 - b.prob_fp if (not b.prob_fp is None) else 1
            elif self.image and hasattr(self.image, 'y') and self == self.image.y:
                b.outline_color = FALSE_POSITIVE_COLOR if b.a is None else TRUE_POSITIVE_COLOR
                b.fill_color = FINISHED_COLOR if self.image.finished else None
                b.alpha = FILL_ALPHA if self.image.finished else 1
            elif self.image and hasattr(self.image, 'y_gt') and self == self.image.y_gt:
                b.outline_color = GT_COLOR
                b.fill_color = None
                b.alpha = 1

    def loss(self, y, fp_loss=1, fn_loss=1, thresh=.5):
        """ Compute the loss between this annotation and y
        """
        self.match_to(y)
        y.colorize()
        loss = 0
        matches = [0] * len(y.bboxes)
        # BUG: should this be above `self.match_to(y)`?
        old_matches = [b.a for b in self.bboxes]
        for b in self.bboxes:
            # Matched and the IoU is greater than 0.5, so this is a true positive
            if (b.a is not None) and b.dist(y.bboxes[b.a]) <= thresh:
                matches[b.a] += 1
            # The bbox is big enough and it didn't match, so this is a false positive
            elif (b.x2 - b.x) > MIN_WIDTH and (b.a is None or y.bboxes[b.a].x2 - y.bboxes[b.a].x > MIN_WIDTH):
                loss += fp_loss
        for i in range(len(y.bboxes)):
            # Nothing matched the given bbox, so this is a false negative
            if matches[i] == 0 and y.bboxes[i].x2 - y.bboxes[i].x > MIN_WIDTH:
                loss += fn_loss
        for i in range(len(self.bboxes)):
            self.bboxes[i].a = old_matches[i]

        # print "loss() i=" + str(self.image.id) + " w=" + str(self.worker.id if self.worker else 'null') + " " + str([b.a for b in self.bboxes])
        return loss

    def estimate_parameters(self, avoid_if_finished=False):
        if avoid_if_finished and self.image.finished:
            return
        self.prob_fn = self.worker.prob_fn
        for b in self.bboxes:
            if not self.worker.is_computer_vision or self.image.params.naive_computer_vision:
                # naive algorithm models CV as a worker, the better algorithm uses its detection confidence to estimate prob_fp
                b.prob_fp = self.worker.prob_fp
            if (not b.a is None) and self.image.params.learn_image_params:
                d2 = b.dist2(self.image.y.bboxes[b.a])
                pi = math.exp(-.5 * d2 / (self.image.sigmas[b.a]**2)) / (math.sqrt(2 * math.pi) * self.image.sigmas[b.a])
                pw = math.exp(-.5 * d2 / (self.worker.sigma**2)) / (math.sqrt(2 * math.pi) * self.worker.sigma)
                b.w = pw / (pi + pw)
                b.sigma = math.sqrt((1 - b.w) * (self.image.sigmas[b.a]**2) + b.w * (self.worker.sigma**2))
            else:
                b.sigma, b.w = self.worker.sigma, 1
        return 0

    def parse(self, data):
        super(CrowdLabelBBox, self).parse(data)
        self.bboxes = []
        for i in range(len(data['bboxes'])):
            b = SingleBBox()
            for k in data['bboxes'][i]:
                setattr(b, k, data['bboxes'][i][k])
            self.image_width, self.image_height = b.image_width, b.image_height
            self.bboxes.append(b)

    def encode(self):
        enc = super(CrowdLabelBBox, self).encode()
        enc['bboxes'] = []
        for i in range(len(self.bboxes)):
            b = {}
            for k in self.bboxes[i].__dict__:
                b[k] = self.bboxes[i].__dict__[k]
            enc['bboxes'].append(b)
        return enc


class SingleBBox:
    def __init__(self, x=None, y=None, x2=None, y2=None, prob_fp=None):
       # bounding box in image coordinates
        self.x = x
        self.y = y
        self.x2 = x2
        self.y2 = y2
        self.sigma = None  # standard deviation in drawing the box boundaries
        self.a = None  # appicable for worker annotations, index into y.bboxes that this box was matched to
        self.prob_fp = prob_fp
        self.w = .5
        self.gtype = 'bbox'

    def dist(self, bbox):
        """ 1 - IoU
        Max dist = 1
        Min dist = 0
        """
        # Union
        ux = max(bbox.x2, self.x2) - min(bbox.x, self.x)
        uy = max(bbox.y2, self.y2) - min(bbox.y, self.y)
        # Intersection
        ix = max(0, min(bbox.x2, self.x2) - max(bbox.x, self.x))
        iy = max(0, min(bbox.y2, self.y2) - max(bbox.y, self.y))
        # 1 - IoU
        return 1.0 - ix * iy / max(1e-8, float(ux * uy))

    def dist2(self, bbox):
        return self.dist(bbox)**2

    def loss(self, bbox, thresh=.5):
        return float(self.dist(bbox) < thresh)
