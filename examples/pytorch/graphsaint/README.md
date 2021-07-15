# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Liu Tang ([@lt610](https://github.com/lt610))

## Dependencies

- Python 3.7.0
- PyTorch 1.6.0
- NumPy 1.19.2
- Scikit-learn 0.23.2
- DGL 0.5.3

## Dataset

All datasets used are provided by Author's [code](https://github.com/GraphSAINT/GraphSAINT). They are available in [Google Drive](https://drive.google.com/drive/folders/1zycmmDES39zVlbVCYs88JTJ1Wm5FbfLz) (alternatively, [Baidu Wangpan (code: f1ao)](https://pan.baidu.com/s/1SOb0SiSAXavwAcNqkttwcg#list/path=%2F)). Once you download the datasets, you need to rename graphsaintdata to data. Dataset summary("m" stands for multi-label classification, and "s" for single-label.):
| Dataset | Nodes | Edges | Degree | Feature | Classes | Train/Val/Test |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| PPI | 14,755 | 225,270 | 15 | 50 | 121(m) | 0.66/0.12/0.22 |
| Flickr | 89,250 | 899,756 | 10 | 500 | 7(s) | 0.50/0.25/0.25 |

Note that the PPI dataset here is different from DGL's built-in variant.

## Minibatch training

Run with following:
```bash
python train_sampling.py --gpu 0 --dataset ppi --sampler node --node-budget 6000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0
python train_sampling.py --gpu 0 --dataset ppi --sampler edge --edge-budget 4000 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1
python train_sampling.py --gpu 0 --dataset ppi --sampler rw --num-roots 3000 --length 2 --num-repeat 50 --n-epochs 1000 --n-hidden 512 --arch 1-0-1-0 --dropout 0.1
python train_sampling.py --gpu 0 --dataset flickr --sampler node --node-budget 8000 --num-repeat 25 --n-epochs 30 --n-hidden 256 --arch 1-1-0 --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler edge --edge-budget 6000 --num-repeat 25 --n-epochs 15 --n-hidden 256 --arch 1-1-0 --dropout 0.2
python train_sampling.py --gpu 0 --dataset flickr --sampler rw --num-roots 6000 --length 2 --num-repeat 25 --n-epochs 15 --n-hidden 256 --arch 1-1-0 --dropout 0.2
```

## Comparison

* Paper: results from the paper
* Running: results from experiments with the authors' code
* DGL: results from experiments with the DGL example

### F1-micro

#### Random node sampler

| Method | PPI | Flickr |
| --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 |
| Running | 0.9628 | 0.5077 |
| DGL | 0.9618 | 0.4828 |

#### Random edge sampler

| Method | PPI | Flickr |
| --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 |
| Running | 0.9810 | 0.5066 |
| DGL | 0.9818 | 0.5054 |

#### Random walk sampler
| Method | PPI | Flickr |
| --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 |
| Running | 0.9812 | 0.5104 |
| DGL | 0.9818 | 0.5018 |

### Sampling time

#### Random node sampler

| Method | PPI | Flickr |
| --- | --- | --- |
| Sampling(Running) | 0.77 | 0.65 |
| Sampling(DGL) | 0.24 | 0.57 |
| Normalization(Running) | 0.69 | 2.84 |
| Normalization(DGL) | 1.04 | 0.41 |

#### Random edge sampler

| Method | PPI | Flickr |
| --- | --- | --- |
| Sampling(Running) | 0.72 | 0.56 |
| Sampling(DGL) | 0.50 | 0.72 |
| Normalization(Running) | 0.68 | 2.62 |
| Normalization(DGL) | 0.61 | 0.38 |

#### Random walk sampler

| Method | PPI | Flickr |
| --- | --- | --- |
| Sampling(Running) | 0.83 | 1.22 |
| Sampling(DGL) | 0.28 | 0.63 |
| Normalization(Running) | 0.87 | 2.60 |
| Normalization(DGL) | 0.70 | 0.42 |
