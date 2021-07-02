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

## Training

We provide training scripts for different datasets.

For training on DeepGlint, one can run

```bash
bash scripts/train_deepglint.sh
```
Deepglint is a large-scale dataset, we randomly select 10% of the classes to construct a subset to train.

For training on full iNatualist dataset, one can run

```bash
bash scripts/train_inat.sh
```

For training on re-bsampled iNatualist dataset, one can run

```bash
bash scripts/train_inat_resampled_1_in_6_per_class.sh
```
We sample a subset of the full iNat2018-Train to attain a drastically different train-time cluster size distribution as iNat2018-Test, which is named as inat_resampled_1_in_6_per_class.

## Inference

In the paper, we have two experiment settings: Clustering with Seen Test Data Distribution and Clustering with Unseen Test Data Distribution.

For Clustering with Seen Test Data Distribution, one can run

```bash
bash scripts/test_deepglint_imbd_sampled_as_deepglint.sh

bash scripts/test_inat.sh
```

**Clustering with Seen Test Data Distribution Performance**
|                    |              IMDB-Test-SameDist |                   iNat2018-Test |
| ------------------ | ------------------------------: | ------------------------------: |
|                 Fp |                           0.793 |                           0.330 |
|                 Fb |                           0.795 |                           0.350 |
|                NMI |                           0.947 |                           0.774 |



For Clustering with Unseen Test Data Distribution, one can run

```bash
bash scripts/test_deepglint_hannah.sh

bash scripts/test_deepglint_imdb.sh

bash scripts/test_inat_train_on_resampled_1_in_6_per_class.sh
```

**Clustering with Seen Test Data Distribution Performance**
|                    |                          Hannah |                            IMDB |                   iNat2018-Test |
| ------------------ | ------------------------------: | ------------------------------: | ------------------------------: |
|                 Fp |                           0.720 |                           0.765 |                           0.294 |
|                 Fb |                           0.700 |                           0.796 |                           0.352 |
|                NMI |                           0.810 |                           0.953 |                           0.764 |

