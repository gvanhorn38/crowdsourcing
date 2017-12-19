# crowdsourcing
Crowdsourcing tools to construct datasets.

# Dataset Format

## Classification
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
    anno : {
        label :
    }
}

gt_label : {
    image_id :
    label : {
        label :
    }
}
```

