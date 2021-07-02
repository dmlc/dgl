Learning Hierarchical Graph Neural Networks for Image Clustering
================================================================

This folder contains the official code for "Learning Hierarchical Graph Neural Networks for Image Clustering"(link needed).

## Setup

Besides DGL (>=0.5.2) and PyTorch, we depend on
- [faiss](https://github.com/facebookresearch/faiss) for K-Nearest Neighbor search.
- [clustering-benchmark](https://github.com/yjxiong/clustering-benchmark) for some of the evaluation metrics.

To set up faiss properly, please refer to their [install instruction](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md). Note that the CUDA version should be aligned with DGL.

## Data

The datasets used for training and test are hosted by several services.

[AWS S3](https://dgl-data.s3.us-west-2.amazonaws.com/dataset/hilander/data.tar.gz) | [Google Drive](https://drive.google.com/file/d/1KLa3uu9ndaCc7YjnSVRLHpcJVMSz868v/view?usp=sharing) | [BaiduPan](https://pan.baidu.com/s/11iRcp84esfkkvdcw3kmPAw) (pwd: wbmh)

After download, unpack the pickled files into `data/`.

## Reproduce Training

We provide training scripts for different datasets.

For full-graph training on DeepFashion, one can run

```bash
bash script/train_df.sh
```

For mini-batch graph training on DeepGlint, one can run

```bash
bash script/train_deepglint.sh
```

## Reproduce Inference

For full-graph inference on DeepFashion, one can run

```bash
bash script/test_df.sh
```

For mini-batch graph inference on Hannah, one can run

```bash
bash script/test_deepglint_hannah.sh
```
