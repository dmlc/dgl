Learning Hierarchical Graph Neural Networks for Image Clustering
================================================================

This folder contains the official code for "Learning Hierarchical Graph Neural Networks for Image Clustering"(link needed).

## Setup

Besides DGL (>=0.5.2) and PyTorch, we depend on [faiss](https://github.com/facebookresearch/faiss) for K-Nearest Neighbor search.

Please refer to their [install instruction](https://github.com/facebookresearch/faiss/blob/master/INSTALL.md) to set up. Note that the CUDA version should be aligned with DGL.

## Data

The datasets used for training and test are hosted by several services.

AWS S3(link needed) | Google Drive(link needed) | BaiduPan(link needed)

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
