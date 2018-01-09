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
    annos : [anno],
    gt_labels : [gt_label]
}
worker : {
    
}

image : {

}

anno : {
    image_id :
    worker_id :
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
    ...
    anno : {
        label : int or str
    }
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
    ...
    bboxes : [bbox]
}

bbox : {
    x : float
    y : float
    x2 : float
    y2 : float
    image_height : int
    image_width : int
}
```
