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
