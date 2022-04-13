# Baseline Code for MAG240M

The code is ported from the R-GAT examples [here](https://github.com/snap-stanford/ogb/tree/master/examples/lsc/mag240m). Please refer to the [OGB-LSC paper](https://arxiv.org/abs/2103.09430) for the detailed setting.

## Installation Requirements

```
ogb>=1.3.0
torch>=1.7.0
```

## Running Preprocessing Script

```
python preprocess.py \
    --rootdir . \
    --author-output-path ./author.npy \
    --inst-output-path ./inst.npy \
    --graph-output-path ./graph.dgl \
    --graph-as-homogeneous \
    --full-output-path ./full.npy
```

This will give you the following files:

* `author.npy`: The author features, preprocessed by averaging the neighboring paper features.
* `inst.npy`: The institution features, preprocessed by averaging the neighboring author features.
* `graph.dgl`: The *homogenized* DGL graph stored in CSC format, which is friendly for neighbor sampling.
  Edge types are stored on the edges as an `int8` feature.  Nodes are in the order of author, institution,
  and paper.
* `full.npy`: The concatenated author, institution, and paper features.

Since that will usually take a long time, we also offer the above files for download:

* [`author.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/author.npy)
* [`inst.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/inst.npy)
* [`graph.dgl`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/graph.dgl)
* [`full.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/full.npy)

In addition, we offer

* [`full_feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/full_feat.npy): The preprocessed full feature matrix
  for running OGB's own baseline. Note that the features are concatenated in the order of paper, author, and
  institution, unlike the one in our baseline code.  It is also preprocessed in float32 arithmetics instead
  of float16 arithmetics.

## Running Training Script

```
python train.py \
    --rootdir . \
    --graph-preprocess-path ./graph.dgl \
    --full-preprocess-path ./full.npy
```

The validation accuracy is 0.701.  We do not have ground truth test labels so we do not report
test accuracy.

## Hardware configurations

We successfully run 8 experiments in parallel on an AWS p4d.24x large instance with the preprocessed feature
matrices stored on an NVMe SSD to enable fast disk read.  Each experiment requires less than 128GB CPU
memory and less than 12GB GPU memory to run.  Every epoch takes around 6 minutes 30 seconds to train and
1 minutes 40 seconds to validate.

If your hard drive is slow, it is best to load all the features into memory for a reasonable training speed.
The CPU memory consumption will go up to as large as 512GB though.
