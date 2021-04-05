# GraphSAINT

This DGL example implements the sampling method proposed in the paper [GraphSAINT: Graph Sampling Based Inductive Learning Method](https://arxiv.org/abs/1907.04931).

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Liu Tang ([@lt610](https://github.com/lt610))

## Dependecies

- Python 3.7.0
- PyTorch 1.6.0
- numpy 1.19.2
- dgl 0.5.3

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available on [Google Drive link](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [BaiduYun link (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Dataset summary("m" stands for multi-class classifification, and "s" for single-class.):
| Dataset | Nodes | Edges | Degree | Feature | Classes | Train/Val/Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) | 0.66/0.12/0.22 |
| Flickr | 89250 | 899756 | 10 | 500 | 7(s) | 0.50/0.25/0.25 |
| Reddit | 232965 | 11606919 | 50 | 602 | 41(s) | 0.66/0.10/0.24 |
| Yelp | 716847 | 6877410 | 10 | 300 | 100(m) | 0.75/0.10/0.15 |
| Amazon | 1598960 | 132169734 | 83 | 200 | 107(m) | 0.85/0.05/0.10 |

## Full graph training

Run with following:
```bash
python train_full.py --gpu 0 --dataset ppi --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm
python train_full.py --gpu 0 --dataset flickr --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_full.py --gpu -1 --dataset reddit --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_full.py --gpu -1 --dataset yelp --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_full.py --gpu -1 --dataset amazon --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
```

## Minibatch training

Run with following:
```bash
python train_sampling.py --gpu 0 --dataset ppi --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm
python train_sampling.py --gpu 0 --dataset ppi --sampler edge --edge-budget 4000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset ppi --sampler rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset flickr --sampler node --node-budget 8000 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler edge --edge-budget 6000 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler rw --num-roots 6000 --length 2 --num-repeat 25 --n-epochs 100 --n-hidden 256 --arch 1-1-0 --batch-norm --dropout 0.2
python train_sampling.py --gpu 0 --dataset reddit --sampler node --node-budget 8000 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler edge --edge-budget 6000 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset reddit --sampler rw --num-roots 2000 --length 4 --num-repeat 50 --n-epochs 100 --n-hidden 128 --arch 1-0-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler node --node-budget 5000 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler edge --edge-budget 2500 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset yelp --sampler rw --num-roots 1250 --length 2 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler node --node-budget 4500 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler edge --edge-budget 2000 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
python train_sampling.py --gpu 0 --dataset amazon --sampler rw --num-roots 1500 --length 2 --num-repeat 50 --n-epochs 100 --n-hidden 512 --arch 1-1-0 --batch-norm --dropout 0.1
```

## Comparison

"Paper" means the results reported in paper. "Running" means the results of runing author's code. DGL means the results of the DGL implementation.

### F1-micro

#### Full graph
| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Full | 0.8765 | 0.5096 | 0.9456 | 0.4253 | 0.2373 |
#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |
| Running | 0.9628 | 0.5077 | 0.9622 | 0.6393 | error |
| DGL | 0.5257 | 0.4943 | 0.8721 | 0.4265 | 0.3906 |
| DGL(Full) | 0.8765 | 0.5096 | 0.9456 | 0.4253 | 0.2373 |

#### Random edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003 | 0.807±0.001 |
| Running | 0.9810 | 0.5066 | 0.9656 | 0.6531 | 0.8071 |
| DGL | 0.9147 | 0.5013 | 0.9243 | 0.4121 | exceed |
| DGL(Full) | 0.8765 | 0.5096 | 0.9456 | 0.4253 | 0.2373 |

#### Random walk sampler
| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |
| Running | 0.9812 | 0.5104 | 0.9648 | 0.6527 | 0.8131 |
| DGL | 0.9199 | 0.5045 | 0.8775 | 0.4078 | 0.3766 |
| DGL(Full) | 0.8765 | 0.5096 | 0.9456 | 0.4253 | 0.2373 |

### Sampling time

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.0139 | 0.9574 | 9.0769 | 30.7790 | error |
| DGL | 0.8725 | 1.1420 | 46.5929 | 68.4477 | 1030.8212 |

#### Random edge sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 0.8712 | 0.8764 | 4.7546 | 17.1285 | 103.6099 |
| DGL | 0.8635 | 1.0033 | 87.5684 | 250.0589 | exceed |

#### Random walk sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.0880 | 1.7588 | 7.2055 | 25.0617 | 172.1458 |
| DGL | 0.7270 | 0.8973 | 58.1987 | 81.8309 | 2918.3490 |