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
    anno : {}
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
