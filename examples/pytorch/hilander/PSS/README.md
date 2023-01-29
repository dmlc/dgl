# PSS

Code for the ECCV '22 submission "PSS: Progressive Sample Selection for Open-World Visual Representation Learning".

## Dependencies

We use python 3.7. The CUDA version needs to be 10.2. Besides DGL==0.6.1, we depend on several packages. To install dependencies using conda:

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

## Training
Run `bash train.sh` to train the model.

## Test
Run `bash test.sh` to evaluate on the test set.