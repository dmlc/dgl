# PSS

Code for the ECCV '22 submission "PSS: Progressive Sample Selection for Open-World Visual Representation Learning".

## Dependencies

We use python 3.7. The CUDA version needs to be 10.2. Besides DGL (>=0.5.2), we depend on several packages. To install dependencies using conda:

```commandline
conda create -n pss python=3.7 # create env
conda activate pss # activate env

conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.2 -c pytorch # install pytorch 1.7 version
conda install -y cudatoolkit=10.2 faiss-gpu=1.6.5 -c pytorch # install faiss gpu version matching cuda 10.2
pip install dgl-cu102 # install dgl for cuda 10.2
pip install tqdm # install tqdm
pip install matplotlib # install matplotlib
pip install pandas # install pandas
pip install pretrainedmodels # install pretrainedmodels
pip install tensorboardX # install tensorboardX
pip install seaborn # install seaborn
pip install scikit-learn
cd ..
git clone https://github.com/yjxiong/clustering-benchmark.git # install clustering-benchmark for evaluation
cd clustering-benchmark
python setup.py install
cd ../PSS
```

## Data

We use the iNaturalist 2018 dataset. 
- download link: https://www.kaggle.com/c/inaturalist-2018/data;
- annotations are in `Smooth_AP/data/Inaturalist`;
- annotation txt files for different data splits are in [S3 link]|[[Google Drive](https://drive.google.com/drive/folders/1xrWogJGef4Ex5OGjiImgA06bAnk2MDrK?usp=sharing)]|[[Baidu Netdisk](https://pan.baidu.com/s/14S0Fns29a4o7kFDlNyyPjA?pwd=uwsg)] (password:uwsg).

Download `train_val2018.tar.gz` and the data split txt files to `data/Inaturalist/` folder. Extract the `tar.gz` files.
The data folder has the following structure:
```bash
PSS
|- data
|  |- Inaturalist
|    |- train2018.json.tar.gz
|    |- train_val2018.tar.gz
|    |- val2018.json.tar.gz
|    |- train_val2018
|    |  |- Actinopterygii
|    |  |- ...
|    |- lin_train_set1.txt
|    |- train_set1.txt
|    |- uin_train_set1.txt
|    |- uout_train_set1.txt
|    |- in_train_set1.txt
|    |- Inaturalist_test_set1.txt
|-...
```

## Pretrained Models
We support ResNet-50 and ViT-Base/16 backbones. For ViT-Base/16 backbone, download [DINO pretrained model](https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth) and use `--pretrained_weights` in training Smooth_AP.

## Training
We provide training scripts for feature extractor Smooth_AP and clustering model Hi-LANDER.
```commandline
# iteration 0
cd Smooth_AP
bash scripts/train_inat.sh # train iter 0 feature extractor
bash scripts/get-features-inat.sh # iter 0 generate features for hilander
cd ../hilander
bash scripts/train_inat.sh # train iter 0 hilander model
bash scripts/test_inat.sh # get iter 0 selected samples and pseudo labels

# iteration 1
cd ../Smooth_AP
bash scripts/finetune_1head_inat.sh # train iter 1 feature extractor
bash scripts/get-features-inat.sh # iter 1 generate features for hilander
cd ../hilander
bash scripts/train_inat.sh # train iter 1 hilander model
bash scripts/test_inat.sh # get iter 1 selected samples and pseudo labels

......
```
Please carefully modify the script files to fit different iteration training. Details of the script file are in the readme files in Smooth_AP and hilander folder.

## Test
Run `bash scripts/test_inat.sh` in Smooth_AP folder.