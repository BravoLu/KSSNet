# Multi-Label-Classification-Baseline

* This repo provides a baseline for multi-label classification task based on resnet101 and KSSNet for coco2014.  

## dataset 

* data/bert_coco.pkl: the word embedding of labels of coco obtained by Bert. 

* data/coco_glove_word2vec.pkl: the word embedding of labels of coco obtained by glove.

  

## Installation

1. create a new conda environment(optional).
```shell
> conda create -n ML python=3.7
> source activate ML 
```

2. install the dependency.

```shell
> cd Multi-Label-Classification-Baseline
> pip install -r requiresments.txt
```

## Train  

```shell
> python train.py --data ${COCO_DIR} --gpu ${GPU_IDS} --ckpt ${CKPT_PATH}
```

## Test

```shell
> python test.py --data ${COCO_DIR} --ckpt ${CHECKPOINT_FILE} --test --gpu ${GPU_IDS}
```

## Results

| model  | mAP   |
| ------ | ----- |
| ResNet | 76\.1 |
| KSSNet | 77\.0 |
