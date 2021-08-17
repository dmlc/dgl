# GraphSAINT

This DGL example implements the paper: GraphSAINT: Graph Sampling Based Inductive Learning Method.

Paper link: https://arxiv.org/abs/1907.04931

Author's code: https://github.com/GraphSAINT/GraphSAINT

Contributor: Jiahang Li ([@ljh1064126026](https://github.com/ljh1064126026))  Tang Liu ([@lt610](https://github.com/lt610))

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

## Config

- The config file is `config.py`.
- Please refer to `sampler.py` to see explanations of some key parameters.

## Minibatch training

Run with following:
```bash
python train_sampling.py --task $task
# e.g. python train_sampling.py --task ppi_n
```

- `$task` includes `ppi_n, ppi_e, ppi_rw, flickr_n, flickr_e, flickr_rw, reddit_n, reddit_e, reddit_rw, yelp_n, yelp_e, yelp_rw, amazon_n, amazon_e, amazon_rw`. For example, `ppi_n` represents running experiments on dataset `ppi` with `node sampler`

## Comparison

* Paper: results from the paper
* Running: results from experiments with the authors' code
* DGL: results from experiments with the DGL example

> Note that we implement offline sampling and online sampling in training phase. Offline sampling means all subgraphs utilized in training phase come from pre-sampled subgraphs. Online sampling means we deprecate all pre-sampled subgraphs and re-sample new subgraphs in training phase.

> Note that the sampling method in pre-sampling phase must be offline sampling.

### F1-micro

#### Random node sampler

| Method | PPI | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.960±0.001 | 0.507±0.001 | 0.962±0.001 | 0.641±0.000 | 0.782±0.004 |
| Running | 0.9628 | 0.5077 |  |  |  |
| DGL_offline |             |             |  |  |  |
| DGL_online |  |  |  |  |  |

#### Random edge sampler

| Method      | PPI         | Flickr      | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.007 | 0.510±0.002 | 0.966±0.001 | 0.653±0.003 | 0.807±0.001 |
| Running | 0.9810 | 0.5066 |  |  |  |
| DGL_offline |             |             |  |  |  |
| DGL_online |  |  |  |  |  |

#### Random walk sampler
| Method      | PPI         | Flickr      | Reddit      | Yelp        | Amazon      |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.981±0.004 | 0.511±0.001 | 0.966±0.001 | 0.653±0.003 | 0.815±0.001 |
| Running | 0.9812 | 0.5104 |             |             |             |
| DGL_offline |             |             |             |             |             |
| DGL_online |  |  |             |             |             |

### Sampling time

- Here sampling time includes consumed time of pre-sampling subgraphs and calculating normalization coefficients in the beginning.

#### Random node sampler

| Method  | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.46 | 3.49 |  |  |  |
| DGL |  |  |  |  |  |

#### Random edge sampler

| Method  | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.4 | 3.18 |  |  |  |
| DGL |  |  |  |  |  |

#### Random walk sampler

| Method  | PPI  | Flickr | Reddit | Yelp | Amazon |
| --- | --- | --- | --- | --- | --- |
| Running | 1.7 | 3.82 |  |  |  |
| DGL |  |  |  |  |  |
