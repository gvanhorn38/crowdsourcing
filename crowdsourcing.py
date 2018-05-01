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

from __future__ import absolute_import, division, print_function

from collections import OrderedDict
import datetime
import json
import os

import numpy as np

class CrowdImage(object):
    """ An image to be annotated.
    """

    def __init__(self, id_, params):
        self.id = id_
        self.params = params
        self.y = None      # Predicted label
        self.x = None      # Image for computer vision
        self.z = OrderedDict() # Worker id to CrowdLabel
        self.d = None      # difficulty parameters
        self.finished = False
        self.workers = []  # list of worker ids
        self.cv_pred = None
        self.encode_exclude = {
            'y': True,
            'y_gt': True,
            'z': True,
            'cv_pred': True,
            'params': True,
            'encode_exclude': True
        }

        # Set when computing the error in `CrowdDataset.compute_error()`
        self.y_gt = None
        self.loss = None

        # Set in `copy_parameters_from`
        self.url = None
        self.fname = None

    def crowdsource_simple(self, avoid_if_finished=False):
        #pylint: disable=unused-argument
        return

    def compute_log_likelihood(self):
        return 0

    def predict_true_labels(self, avoid_if_finished=False):
        #pylint: disable=unused-argument
        return

    def estimate_parameters(self, avoid_if_finished=False):
        #pylint: disable=unused-argument
        return

    def check_finished(self, set_finished=True):
        #pylint: disable=unused-argument
        return

    def filename(self):
        return os.basename(self.url)

    def num_annotations(self):
        num = 0
        if not self.z is None:
            if self.params.cv_worker and self.params.cv_worker.id in self.z:
                has_cv = 1
            else:
                has_cv = 0
            num = len(self.z) - has_cv
        return num

    def parse(self, data):
        for k in data:
            setattr(self, k, data[k])

    def encode(self):
        data = {}
        for k in self.__dict__:
            if k not in self.encode_exclude:
                data[k] = self.__dict__[k]
        return data

    def copy_parameters_from(self, image, full=True):
        #pylint: disable=unused-argument
        if hasattr(image, 'url'):
            self.url = image.url
        if hasattr(image, 'fname'):
            self.fname = image.fname


class CrowdWorker(object):
    """ A worker.
    """

    def __init__(self, id_, params):
        self.id = id_
        self.params = params
        # set of images annotated by this worker, image id to CrowdImage
        self.images = {}
        self.finished = False
        self.encode_exclude = {
            'images': True,
            'params': True,
            'encode_exclude': True
        }

        # set in `CrowdDataset.get_computer_vision_probabilities()`
        self.is_computer_vision = False

    def compute_log_likelihood(self):
        return 0

    def estimate_parameters(self, avoid_if_finished=False):
        #pylint: disable=unused-argument
        return 0

    def parse(self, data):
        for k in data:
            setattr(self, k, data[k])

    def encode(self):
        data = {}
        for k in self.__dict__:
            if k not in self.encode_exclude:
                data[k] = self.__dict__[k]
        return data

    def copy_parameters_from(self, worker, full=True):
        #pylint: disable=unused-argument
        return


class CrowdLabel(object):
    """ An annotation.
    """

    def __init__(self, image, worker):
        self.image = image  # CrowdImage
        self.worker = worker  # CrowdWorker
        self.encode_exclude = {
            'worker': True,
            'image': True,
            'encode_exclude': True,
            'raw_data': True
        }

        # Is this a computer vision annotation?
        # TODO: refactor to a property
        self._is_computer_vision = (self.worker and
                                    self.worker.is_computer_vision)

        # Assigned in `parse`
        self.raw_data = None

    def compute_log_likelihood(self):
        return 0

    def loss(self, y):
        #pylint: disable=unused-argument
        return 1

    def copy_into(self, into):
        for attr in dir(self):
            if (not callable(attr) and not attr.startswith("__") and
                    attr != "image" and attr != "worker"):
                setattr(into, attr, getattr(self, attr))

    def estimate_parameters(self):
        return 0

    # TODO: refactor to a property
    def is_computer_vision(self):
        # return self.worker and self.worker.is_computer_vision
        return self._is_computer_vision

    def parse(self, data):
        self.raw_data = data
        for k in data:
            setattr(self, k, data[k])

    def encode(self):
        data = {}
        for k in self.__dict__:
            if k not in self.encode_exclude:
                data[k] = self.__dict__[k]
        return data


class CrowdDataset(object):
    """ A dataset, holding images, workers and labels.
    """

    def __init__(self, debug=0, min_risk=0.005, learn_worker_params=True,
                 learn_image_params=True, estimate_priors_automatically=False,
                 computer_vision_predictor=None, naive_computer_vision=False,
                 add_computer_vision_to_workers=True, image_dir=None,
                 name=""):

        self.debug = debug
        self.name = name
        self.min_risk = min_risk
        self.learn_worker_params = learn_worker_params
        self.learn_image_params = learn_image_params
        self.computer_vision_predictor = computer_vision_predictor
        self.image_dir = image_dir
        self.estimate_priors_automatically = estimate_priors_automatically
        self.finished = False
        self.encode_exclude = {
            'workers': True,
            'images': True,
            'cv_worker': True,
            '_CrowdImageClass_': True,
            '_CrowdWorkerClass_': True,
            '_CrowdLabelClass_': True,
            'computer_vision_predictor': True,
            'encode_exclude': True
        }
        self.naive_computer_vision = naive_computer_vision
        self.cv_iter = 0
        self.cv_worker = None  # A CrowdWorker whose CrowdLabels come from the
        # `computer_vision_predictor`
        self.add_computer_vision_to_workers = add_computer_vision_to_workers
        self.images = {}  # image id to CrowdImage
        self.workers = {}  # worker id to CrowdWorker

        # assigned in `copy_parameters_from`
        self.fname = None

    def crowdsource_simple(self, avoid_if_finished=False):
        """ Set the predicted labels to the consensus median.
        """
        for image in self.images.itervalues():
            image.crowdsource_simple(avoid_if_finished=avoid_if_finished)

    def estimate_priors(self, gt_dataset=None):
        """ Estimate priors globally over the whole dataset.
        """
        #pylint: disable=unused-argument
        return

    def initialize_parameters(self, avoid_if_finished=False):
        """ Initialize the parameters of the CrowdWorkers and the CrowdImages.
        """
        #pylint: disable=unused-argument
        return

    def compute_log_likelihood(self):
        ll = 0
        for i in self.images:
            ll += self.images[i].compute_log_likelihood()
        for w in self.workers:
            ll += self.workers[w].compute_log_likelihood()
        for i in self.images:
            for w in self.images[i].z:
                ll += self.images[i].z[w].compute_log_likelihood()
        return ll

    def estimate_parameters(self, max_iters=10, avoid_if_finished=False):
        """Estimate the image labels, and potentially other parameters, for
        this dataset.
        """

        # Have the images assign themselves an estimate of the labels
        self.crowdsource_simple(avoid_if_finished=avoid_if_finished)

        # If we have a computer vision system then get updated labels from it
        if self.computer_vision_predictor is not None:
            self.initialize_parameters(avoid_if_finished=avoid_if_finished)

            # Get updated image labels
            for image in self.images.itervalues():
                image.predict_true_labels(avoid_if_finished=avoid_if_finished)

            # Get CrowdLabels from the computer vision system
            self.get_computer_vision_probabilities()

        # Update our estimate of the dataset wide priors
        if self.estimate_priors_automatically:
            self.estimate_priors()

        # Initialize the parameters of the workers and images
        self.initialize_parameters(avoid_if_finished=avoid_if_finished)

        # Maximum likelihood estimation
        log_likelihood = -np.inf
        old_likelihood = -np.inf
        for it in xrange(max_iters):

            if self.debug > 1:
                print("Estimate params for " + self.name + ", iter " +
                      str(it + 1) + " likelihood=" + str(log_likelihood))

            # Estimate label predictions in each image using worker labels and
            # current worker parameters
            for image in self.images.itervalues():
                image.predict_true_labels(avoid_if_finished=avoid_if_finished)

            # Estimate difficulty parameters for each image
            if self.learn_image_params:
                for image in self.images.itervalues():
                    image.estimate_parameters(
                        avoid_if_finished=avoid_if_finished)

            # Estimate skill parameters for each worker
            if self.learn_worker_params:
                for worker in self.workers.itervalues():
                    worker.estimate_parameters(avoid_if_finished=avoid_if_finished)

            # Estimate response probability parameters for each worker
            for image in self.images.itervalues():
                for label in image.z.itervalues():
                    label.estimate_parameters()

            # Check the new log likelihood of the dataset and finish on
            # convergence
            log_likelihood = self.compute_log_likelihood()
            if log_likelihood <= old_likelihood:
                if self.debug > 1:
                    print("New likelihood=" + str(log_likelihood))
                break
            old_likelihood = log_likelihood

        return log_likelihood

    def get_computer_vision_probabilities(self, method='at_least_one_worker'):
        """ Get updated computer vision probabilities for all images in the
        dataset. This will retrain the computer vision system, and then extract
        CrowdLabels.
        """

        image_ids = [image_id for image_id in self.images]
        images = [self.images[image_id] for image_id in image_ids]
        labels = [self.images[image_id].y for image_id in image_ids]

        # Specify the images that can be used for training
        if method == 'at_least_one_worker':
            valid_train = []
            has_cv = self.cv_worker != None
            for image_id in image_ids:
                has_cv_anno = int(has_cv and
                                  self.cv_worker.id in self.images[image_id].z)
                num_annos = len(self.images[image_id].z)
                if num_annos - has_cv_anno > 0:
                    valid_train.append(True)
                else:
                    valid_train.append(False)

        elif method == 'is_finished':
            valid_train = [
                self.images[image_id].finished for image_id in image_ids]

        # Create a new crowd worker and mark them as computer vision
        prev_cv_worker = self.cv_worker
        cv_worker_id = 'computer_vision_iter' + str(self.cv_iter)
        self.cv_worker = self._CrowdWorkerClass_(cv_worker_id, params=self)
        self.cv_worker.is_computer_vision = True
        self.cv_iter += 1

        # Get the CrowdLabels from the computer vision system
        cv_preds = self.computer_vision_predictor.predict_probs(
            images=images,
            labels=labels,
            valid_train=valid_train,
            cache_name=self.fname + '.computer_vision_cache',
            cv_worker=self.cv_worker,
            naive=self.naive_computer_vision
        )

        cv_preds_dict = {label.image.id: label for label in cv_preds}
        cv_preds = [cv_preds_dict[image_id] for image_id in image_ids]

        # Overwrite the computer vision predictions for each image with the new
        # CrowdLabels
        for image_id in image_ids:
            self.images[image_id].cv_pred = cv_preds_dict[image_id]
            self.cv_worker.images[image_id] = self.images[image_id]

        # Add the computer vision CrowdLabels to the label arrays for each
        # CrowdImage
        if self.add_computer_vision_to_workers:

            # Delete the previous cv worker from our worker dict
            if prev_cv_worker:
                del self.workers[prev_cv_worker.id]

            # Add the CrowdLabel to each image
            for image_id in image_ids:

                # Delete the previous computer vision CrowdLabel
                if (prev_cv_worker and
                        prev_cv_worker.id in self.images[image_id].z):
                    del self.images[image_id].z[prev_cv_worker.id]

                # Put the new computer vision CrowdLabel in the image's label
                #  dict
                self.images[image_id].z[self.cv_worker.id] = \
                  cv_preds_dict[image_id]

                # Find the index of the previous computer vision CrowdWorker
                # and replace it with the new computer vision CrowdWorker.
                worker_ind = -1
                for j, worker_id in enumerate(self.images[image_id].workers):
                    if prev_cv_worker and worker_id == prev_cv_worker.id:
                        worker_ind = j
                        break
                if worker_ind >= 0:
                    self.images[image_id].workers[worker_ind] = \
                      self.cv_worker.id
                else:
                    self.images[image_id].workers.append(self.cv_worker.id)

            self.workers[self.cv_worker.id] = self.cv_worker

    def check_finished_annotations(self, set_finished=True):
        """Return a dict mapping image id to a bool indicating whether the
        image is finished.
        """
        finished = {}
        for image_id, image in self.images.iteritems():
            finished[image_id] = image.check_finished(
                set_finished=set_finished)
        return finished

    def num_unfinished(self, max_annos=float('Inf'), full_dataset=None):
        """ Return the number of unfinished images.
        "Finished" is either:
          The image is marked as finished.
          The image has exceed the `max_annos` parameter.
          The full_dataset has no more annotations for the image.
        """

        num = 0
        for image_id, image in self.images.iteritems():

            if image.z is None:
                continue

            if image.finished:
                continue

            has_cv = 0
            if self.cv_worker and self.cv_worker.id in image.z:
                has_cv = 1

            fd_has_cv = 0
            if (full_dataset and full_dataset.cv_worker and
                    full_dataset.cv_worker.id in
                    full_dataset.images[image_id].z):
                fd_has_cv = 1

            num_annos_available = max_annos
            if full_dataset is not None:
                num_non_cv_annos = len(
                    full_dataset.images[image_id].z) - fd_has_cv
                num_annos_available = min(max_annos, num_non_cv_annos)

            num_current_annos = len(image.z) - has_cv

            if num_current_annos < num_annos_available:
                num += 1

        return num

    def num_annotations(self):
        """Return the total number of annotations in the dataset.
        """

        num = 0
        for image in self.images.itervalues():
            num += image.num_annotations()
        return num

    def risk(self):
        """Return the average risk across all images in the dataset.
        """
        if len(self.images) == 0:
            return 0

        r = 0.
        for image in self.images.itervalues():
            r += image.risk
        return r / len(self.images)

    def choose_images_to_annotate_next(self, sort_method="num_annos",
                                       full_dataset=None):
        """Return a list of image ids that should be annotated next.
        """

        if sort_method == "num_annos":
            queue = sorted(self.images.items(), key=lambda x: len(x[1].z))
        elif sort_method == "risk":
            def risk_func(image):
                if hasattr(image, "risk"):
                    return -image.risk
                else:
                    return len(image.z)
            queue = sorted(self.images.items(),
                           key=lambda x: risk_func(x[1]))
        elif sort_method == "normalized_risk":
            def norm_risk_func(image):
                if len(image.z) > 0 and hasattr(image, "risk"):
                    return -image.risk / len(image.z)
                else:
                    return len(image.z)
            queue = sorted(self.images.items(),
                           key=lambda x: norm_risk_func(x[1])
                          )
        else:
            queue = full_dataset.images.items()

        image_ids = []
        for iq in queue:
            image_id = iq[0]
            if not self.images[image_id].finished:
                image_ids.append(image_id)
        return image_ids

    def compute_error(self, gt_dataset, **kwds):
        """Get the average error compared to a ground truth dataset.
        """

        err = 0.
        num_images = 0
        for image_id, gt_image in gt_dataset.images.iteritems():

            image = self.images[image_id]

            # Get the ground truth label
            if hasattr(gt_image, 'y_gt') and gt_image.y_gt is not None:
                y_gt = gt_image.y_gt
            else:
                y_gt = gt_image.y

            if y_gt is not None:
                # Compute the loss to the current predicted label
                image.loss = image.y.loss(y_gt, **kwds)
                err += image.loss
                num_images += 1

        if num_images == 0:
            return 0.

        return err / float(num_images)

    def load(self, fname, max_assignments=None, sort_annos=False,
             overwrite_workers=True, load_dataset=True, load_workers=True,
             load_images=True, load_annos=True, load_gt_annos=True,
             load_combined_labels=True):
        """ Load in a dataset json file.
        Args:
          sort_annos (bool): Should the annotations be sorted by timestamp?
          overwrite_workers (bool): Should worker instances be reinitialized?
            (e.g. maybe we are loading in a new data after learning worker
            skill parameters).
        """
        self.fname = fname
        with open(fname) as f:
            data = json.load(f)
        #self.images = {} #NOTE: I don't think we want to reset the images here.
        if overwrite_workers:
            self.workers = {}
        if 'dataset' in data and load_dataset:
            self.parse(data['dataset'])
        if 'workers' in data and load_workers:
            for w in data['workers']:
                if w not in self.workers:
                    self.workers[w] = self._CrowdWorkerClass_(w, self)
                    self.workers[w].parse(data['workers'][w])
        if 'images' in data and load_images:
            for i in data['images']:
                self.images[i] = self._CrowdImageClass_(i, self)
                self.images[i].parse(data['images'][i])
        if 'annos' in data and load_annos:
            annos = data['annos']

            # Sort the annotations by the 'created_at' field
            if sort_annos:
                for anno in annos:
                    try:
                        t = datetime.datetime.strptime(anno['created_at'],
                                                       '%Y-%m-%d %H:%M:%S.%f')
                    except ValueError:
                        t = datetime.datetime.strptime(anno['created_at'],
                                                       '%Y-%m-%d %H:%M:%S')
                    anno['time'] = t
                annos.sort(key=lambda x: x['time'])
                for anno in annos:
                    del anno['time']

            for l in annos:
                i, w, a = l['image_id'], l['worker_id'], l['anno']
                if i not in self.images:
                    self.images[i] = self._CrowdImageClass_(i, self)
                if w not in self.workers:
                    self.workers[w] = self._CrowdWorkerClass_(w, self)
                if self.cv_worker and self.cv_worker.id in self.images[i].z:
                    has_cv = 1
                else:
                    has_cv = 0
                if (max_assignments is None or
                        len(self.images[i].z) - has_cv < max_assignments):
                    z = self._CrowdLabelClass_(self.images[i], self.workers[w])
                    z.parse(a)
                    self.images[i].z[w] = z
                    self.images[i].workers.append(w)
                    self.workers[w].images[i] = self.images[i]

        # Are there ground truth labels available?
        if 'gt_labels' in data and load_gt_annos:
            for l in data['gt_labels']:
                i, a = l['image_id'], l['label']
                self.images[i].y_gt = self._CrowdLabelClass_(
                    self.images[i], None)
                self.images[i].y_gt.parse(a)
                self.images[i].y = self.images[i].y_gt

        # Are there combined labels available?
        if 'combined_labels' in data and load_combined_labels:
            for l in data['combined_labels']:
                i, a = l['image_id'], l['label']
                self.images[i].y = self._CrowdLabelClass_(self.images[i], None)
                self.images[i].y.parse(a)

    def save(self, fname, save_dataset=True, save_images=True, save_workers=True,
             save_annos=True, save_gt_labels=True, save_combined_labels=True):
        """Save a dataset as a json file.
        """
        data = {}
        if save_dataset:
            data['dataset'] = self.encode()
        if save_images:
            data['images'] = {}
            for i in self.images:
                data['images'][i] = self.images[i].encode()
        if save_workers:
            data['workers'] = {}
            for w in self.workers:
                data['workers'][w] = self.workers[w].encode()
        if save_annos:
            data['annos'] = []
            for i in self.images:
                for w in self.images[i].z:
                    data['annos'].append({
                        'image_id': i,
                        'worker_id': w,
                        'anno': self.images[i].z[w].encode()
                    })
        if save_gt_labels:
            data['gt_labels'] = []
            for i in self.images:
                if hasattr(self.images[i], 'y_gt') and self.images[i].y_gt:
                    data['gt_labels'].append({
                        'image_id': i,
                        'label': self.images[i].y_gt.encode()
                    })
        if save_combined_labels:
            data['combined_labels'] = []
            for i in self.images:
                if hasattr(self.images[i], 'y') and self.images[i].y:
                    data['combined_labels'].append({
                        'image_id': i,
                        'label': self.images[i].y.encode()
                    })
        with open(fname, 'w') as f:
            json.dump(data, f)

    def parse(self, data):
        for k in data:
            setattr(self, k, data[k])

    def encode(self):
        data = {}
        for k in self.__dict__:
            if not k in self.encode_exclude:
                data[k] = self.__dict__[k]
        return data

    def scan_image_directory(self, dir_name):
        print('Scanning images from ' + dir_name + '...')
        images = [f for f in os.listdir(dir_name)
                  if os.path.isfile(os.path.join(dir_name, f))]
        if not hasattr(self, 'images'):
            self.images = {}
        if not hasattr(self, 'workers'):
            self.workers = {}
        for f in images:
            i, _ = os.path.splitext(f)
            self.images[i] = self._CrowdImageClass_(i, self)
            self.images[i].fname = os.path.join(dir_name, f)

    def copy_parameters_from(self, dataset, full=True):
        #pylint: disable=unused-argument
        if hasattr(dataset, 'fname'):
            self.fname = dataset.fname
