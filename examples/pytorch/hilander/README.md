Learning Hierarchical Graph Neural Networks for Image Clustering
================================================================

This folder contains the official code for [Learning Hierarchical Graph Neural Networks for Image Clustering](https://arxiv.org/abs/2107.01319).

## Setup

We use python 3.7. The CUDA version needs to be 10.2. Besides DGL (>=0.8), we depend on several packages. To install dependencies using conda:
```bash
conda create -n Hilander # create env
conda activate Hilander # activate env
conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch # install pytorch 1.7 version
conda install -y cudatoolkit=10.2 faiss-gpu=1.6.5 -c pytorch # install faiss gpu version matching cuda 10.2
pip install dgl-cu102 dglgo -f https://data.dgl.ai/wheels/repo.html # install the latest dgl for cuda 10.2
pip install tqdm # install tqdm
git clone https://github.com/yjxiong/clustering-benchmark.git # install clustering-benchmark for evaluation
cd clustering-benchmark
python setup.py install
cd ../
```

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

For training on re-sampled iNatualist dataset, one can run

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
|                 Fp |                           0.779 |                           0.330 |
|                 Fb |                           0.819 |                           0.350 |
|                NMI |                           0.949 |                           0.774 |
* The results might fluctuate a little due to the randomness introduced by gpu knn building using faiss-gpu.


For Clustering with Unseen Test Data Distribution, one can run

```bash
bash scripts/test_deepglint_hannah.sh

bash scripts/test_deepglint_imdb.sh

bash scripts/test_inat_train_on_resampled_1_in_6_per_class.sh
```

**Clustering with Unseen Test Data Distribution Performance**
|                    |                          Hannah |                            IMDB |                   iNat2018-Test |
| ------------------ | ------------------------------: | ------------------------------: | ------------------------------: |
|                 Fp |                           0.741 |                           0.717 |                           0.294 |
|                 Fb |                           0.706 |                           0.810 |                           0.352 |
|                NMI |                           0.810 |                           0.953 |                           0.764 |
* The results might fluctuate a little due to the randomness introduced by gpu knn building using faiss-gpu.

