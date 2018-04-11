# crowdsourcing
Crowdsourcing tools to construct datasets.

# Dataset Format

## Generic Format
```
{
    dataset : {
        # dataset parameters
    },
    workers : {
        worker_id : worker
    },
    images : {
        image_id : image
    }
    annos : [annotation],
    gt_labels : [gt_label]
}
worker : {
    
}

image : {

}

annotation : {
    image_id :
    worker_id :
    anno : {} # custom data for a specific annotation type
}

gt_label : {
    image_id :
    label : {}
}
```

## Classification Format

```
dataset : {
    ...
    'num_classes' : int,
    'taxonomy_data' : taxonomy_data
}

taxonomy_data : [{
    key : int or str,
    parent : int or str,
    data : {}
}]

anno : {
    label : int or str
}


gt_label : {
    ...
    label : {
        label : int or str
    }
}
```

## Detection Format
```
image : {
    ...
    width : int
    height : int
}

anno : {
    bboxes : [bbox]
}

bbox : {
    x : float (left edge in image coordinates)
    y : float (top edge in image coordinates)
    x2 : float (right edge in image coordinates)
    y2 : float (bottom edge in image coordinates)
    image_height : int
    image_width : int
}
```
(x, y) is the top left corner of the bounding box, and (x2, y2) is the bottom right corner of the bounding box.


# Merging Bounding Boxes Example

This example demonstrates how to merge redundant bounding boxes to produce a dataset that can be used to train a detector (among other things). This is a "static" usage of this code base, in the sense that all of your annotations are collected, and you are looking to merge the redundant boxes to produce a "finished" dataset. This example makes use of the Annotation Tools](https://github.com/visipedia/annotation_tools) repo for visualizing the dataset.

## Dataset Format

You need to have your annotations in the following JSON format:
```python
{
    "dataset" : {},
    "images" : {
        "0" : {
            "height" : 600,
            "width" : 800,
            "url" : "http://localhost:/images/0.jpg"
        },
        ... # more images
    },
    "workers" : {},
    "annos" : [
        {
            "anno" : {
                "bboxes" : [
                    {
                        "image_height" : 600,
                        "image_width" : 800,
                        "x" : 15,
                        "x2" : 330,
                        "y" : 75,
                        "y2" : 400
                    },
                    ... # more boxes
                ]
            },
            "image_id" : "0",
            "worker_id" : "XYZ"
        },
        ... # more annotations
    ]   
}
```
A few notes:
  * The `dataset` dictionary can be empty, but must be present. 
  * You should have an entry for each image in the `images` dictionary where the key is the image id and the value is a dictionary containing the `width` and `height` of the image. The `url` field is not required, but it is convenient for visualizing the dataset. 
  * The `workers` dictionary can be empty, but must be present. The code will create missing workers based on the data in the `annos` list. You can also populate this dictionary with worker specific information if it is convenient.
  * Because image ids and worker ids are used as keys in a JSON dictionary, they must be strings. Make sure to store string values for the `image_id` and `worker_id` fields when creating the `annos` list. 
  * The `annos` field is a list, as opposed to a dictionary. Each item in the `annos` list represent the annotations completed by a single worker for a single image. The image id is stored in `image_id`, the worker id is stored in `worker_id` and the list of bounding boxes drawn by the worker are stored in `bboxes` in the `anno` dictionary. The `bboxes` should be a list of dictionaries, where each dictionary represents a single box that was drawn. If no boxes were drawn by the worker, then `bboxes` should be an empty list (i.e. `bboxes : []`). The values of `image_height` and `image_width` should be the same values that are stored in the `width` and `height` fields for the corresponding image (this is redundant, but currently necessary). The top left corner of the box is represented with `x` and `y`, and the bottom right corner of the box is represent with `x2` and `y2`. The box coordinates should be in pixel space. The origin is the upper left hand corner.  

I will assume that you have constructed the dataset and saved it. I'll assume that the `RAW_DATASET` variable holds the path to the dataset.

## Visualize the Raw Dataset

It is recommended to visualize the raw worker annotations to ensure that you constructed the dataset format correctly. We will convert the raw dataset into a COCO representation and visualize the annotations using the [Annotation Tools](https://github.com/visipedia/annotation_tools) repo. Note that you need to have a `url` field for each image in order to visualize the dataset.

The following python code will convert your dataset to the COCO format and save it to the path stored in the `RAW_COCO_DATASET` variable:

```python
import json

RAW_DATASET = '...' # Path to the crowdsourcing dataset
RAW_COCO_DATASET = '...' # Path to save the COCO dataset

with open(RAW_DATASET) as f:
    raw_dataset = json.load(f)

default_category_id = 0
annotations = []
anno_id = 0
for anno in raw_dataset['annos']:
    image_id = anno['image_id']
    for bbox in anno['anno']['bboxes']:
        x = bbox['x']
        y = bbox['y']
        x2 = bbox['x2']
        y2 = bbox['y2']

        w = x2 - x
        h = y2 - y

        annotations.append({
            'image_id' : image_id,
            'category_id' : default_category_id,
            'bbox' : [x, y, w, h],
            'id' : anno_id
        })
        anno_id += 1

coco_images = []
for image_id, image in raw_dataset['images'].iteritems():
    coco_images.append({
        'id' : image_id,
        'url' : image['url'],
        'width' : image['width'],
        'height' : image['height']
    })

categories = [{
    'id' : default_category_id,
    'name' : "Object",
    'supercategory': "Object"
}]

licenses = []
info = {}

coco_dataset = {
    "categories" : categories,
    "images" : coco_images,
    "annotations" : annotations,
    "licenses" : licenses,
    "info" : info
}

with open(RAW_COCO_DATASET, "w") as f:
    json.dump(coco_dataset, f)
```  

In the `annotation_tools/` repo, we can then load up the coco dataset to visualize the worker annotations:

```
# Clear the database (in case we were working on other annotations before)
python -m annotation_tools.db_dataset_utils \
--action drop

# Load in the dataset
python -m annotation_tools.db_dataset_utils \
--action load \
--dataset $RAW_COCO_DATASET \
--normalize
```

Make sure to start the webserver:
```
python run.py --port 8008
```

Then you can go to `http://localhost:8008/edit_task/?start=0&end=100` to inspect the annotations on the first 100 images (sorted by image ids). Note that this interface does not render different worker's annotations separately, it simply renders them all together, so you should see redundant boxes on each object. We'll use the `CrowdDatasetBBox` class to merge the redundant annotations together.

## Merging the Annotations to Produce a Combined Dataset

At this point we have a dataset in the correct format (stored at `RAW_DATASET`) and we have confirmed that the worker boxes were processed correctly. We can now use the `CrowdDatasetBBox` class to merge the annotations. You'll need to add the path to the crowdsourcing repo to your PYTHONPATH environment variable. 

The following python code will do 4 things:
  1. Loads the raw dataset from `RAW_DATASET` and creates a `CrowdDatasetBBox` instance.
  2. Produces object location priors. 
  3. Estimates the boxes for each image.
  4. Saves the predicted boxes as a combined dataset to `COMBINED_DATASET`.

```python
from crowdsourcing.annotations.detection.bbox import CrowdDatasetBBox

RAW_DATASET = '...' # Path to the crowdsourcing dataset
COMBINED_DATASET = '...' # Path to save the combined dataset

full_dataset = CrowdDatasetBBox(
  debug=0,
  learn_worker_params=True,
  learn_image_params=True,
  estimate_priors_automatically=False,
  computer_vision_predictor=None,
  naive_computer_vision=False,
  min_risk=0.02
)
# Load in the worker annotations
full_dataset.load(RAW_DATASET)

# Generate object location priors using all of the worker annotations
box_set = full_dataset.get_big_bbox_set()

# Estimate the boxes for each image
full_dataset.estimate_parameters(avoid_if_finished=False)

# Get the finished annotations
image_id_to_finished = full_dataset.check_finished_annotations(set_finished=True)

num_finished = sum(image_id_to_finished.values())
print "%d / %d (%0.2f%%) images are finished" % (num_finished, len(image_id_to_finished), 100. * float(num_finished) / len(image_id_to_finished))

# Save the computed boxes
full_dataset.save(
  COMBINED_DATASET,
  save_dataset=True,
  save_images=True,
  save_workers=True,
  save_annos=True,
  save_combined_labels=True
)
```

## Visualize the Combined Dataset

We can convert the combined dataset into a COCO style dataset and visualize the results. We'll store the COCO style dataset to the path store in `COMBINED_COCO_DATASET`. The below code is very similar to the above COCO conversion code, there is just a small change in the fields being accessed. 

```python
import json

COMBINED_DATASET = '...' # Path to the combined dataset
COMBINED_COCO_DATASET = '...' # Path to save the COCO dataset

with open(COMBINED_DATASET) as f:
    combined_dataset = json.load(f)

default_category_id = 0
annotations = []
anno_id = 0
for label in combined_dataset['combined_labels']:
    image_id = label['image_id']
    for bbox in label['label']['bboxes']:
        x = bbox['x']
        y = bbox['y']
        x2 = bbox['x2']
        y2 = bbox['y2']

        w = x2 - x
        h = y2 - y

        annotations.append({
            'image_id' : image_id,
            'category_id' : default_category_id,
            'bbox' : [x, y, w, h],
            'id' : anno_id
        })
        anno_id += 1

coco_images = []
for image_id, image in combined_dataset['images'].iteritems():
    coco_images.append({
        'id' : image_id,
        'url' : image['url'],
        'width' : image['width'],
        'height' : image['height']
    })

categories = [{
    'id' : default_category_id,
    'name' : "Object",
    'supercategory': "Object"
}]

licenses = []
info = {}

coco_dataset = {
    "categories" : categories,
    "images" : coco_images,
    "annotations" : annotations,
    "licenses" : licenses,
    "info" : info
}

with open(COMBINED_COCO_DATASET, "w") as f:
    json.dump(coco_dataset, f)
```

The `CrowdDatasetBBox` also computed the risk for each image and this value was saved in the combined dataset. We can sort the images by risk and visualize the riskiest images. 
```python
image_ids_and_risk = [(image['id'], image['risk']) for image in combined_dataset['images'].values()]
image_ids_and_risk.sort(key=lambda x: x[1])
image_ids_and_risk.reverse()

risky_image_ids = ",".join([x[0] for x in image_ids_and_risk[:100]])
print "Risky Image Visualization URL:"
print "http://localhost:8008/edit_task/?image_ids=%s" % (risky_image_ids,)

```


In the `annotation_tools/` repo, we can then load up the coco dataset to visualize the combined annotations:

```
# Clear the database (to remove the worker annotations loaded up previously)
python -m annotation_tools.db_dataset_utils \
--action drop

# Load in the dataset
python -m annotation_tools.db_dataset_utils \
--action load \
--dataset $COMBINED_COCO_DATASET \
--normalize
```

Make sure the webserver is still running:
```
python run.py --port 8008
```

Then you can go to `http://localhost:8008/edit_task/?start=0&end=100` to inspect the annotations on the first 100 images (sorted by image ids). You should hopefully see one box on each object. You can also visit the url that was printed above to visualize the 100 riskiest images. These images might need to be edited. If you make adjustments to the annotations, then you can export the dataset with:
```
$ python -m annotation_tools.db_dataset_utils \
--action export \
--output $UPDATED_COMBINED_COCO_DATASET \
--denormalize
```
The final version of your dataset will be saved at the path stored in `UPDATED_COMBINED_COCO_DATASET`.