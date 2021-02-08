# DGL Implementation of InfoGraph
This DGL example implements the model proposed in the paper [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/abs/1908.01000).

Paper link: https://arxiv.org/abs/1908.01000

Author's code: https://github.com/fanyun-sun/InfoGraph

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) during his Applied Scientist Intern work at the AWS Shanghai AI Lab.

## Dependecies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.5.3

## Datasets

##### Unsupervised Graph Classification Dataset:

 'MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K' of dgl.data.GINDataset.

| Dataset         | MUTAG | PTC   | RDT-B  | RDT-M5K | IMDB-B | IMDB-M |
| --------------- | ----- | ----- | ------ | ------- | ------ | ------ |
| # Graphs        | 188   | 344   | 2000   | 4999    | 1000   | 1500   |
| # Classes       | 2     | 2     | 2      | 5       | 2      | 3      |
| Avg. Graph Size | 17.93 | 14.29 | 429.63 | 508.52  | 19.77  | 13.00  |


## Arguments

##### 	Unsupervised Graph Classification:

###### Dataset options

```
--dataname          str     The graph dataset name.             Default is 'MUTAG'.
```

###### GPU options

```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Training options

```
--epochs           int     Number of training epochs.             Default is 20.
--batch_size       int     Size of a training batch               Default is 128.
--lr               float   Adam optimizer learning rate.          Default is 0.01.
```

###### Model options

```
--n_layers         int     Number of GIN layers.                  Default is 3.
--hid_dim          int     Dimension of hidden layer.             Default is 32.
```

## How to run examples

Training and testing unsupervised model on MUTAG.(We recommend using cpu)
```bash
# MUTAG:
python unsupervised.py --dataname MUTAG --n_layers 4 --hid_dim 32
```
Replace 'MUTAG' with dataname in [MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'] if you'd like to try other datasets.

## 	Performance

The hyperparameter setting in our implementation is identical to that reported in the paper.

##### Unsupervised Graph Classification:

|      Dataset      | MUTAG |  PTC  | REDDIT-B | REDDIT-M | IMDB-B | IMDB-M |
| :---------------: | :---: | :---: | :------: | -------- | ------ | ------ |
| Accuract Reported | 89.01 | 61.65 |  82.50   | 53.46    | 73.03  | 49.69  |
|  This repository  | 89.88 | 63.54 |  88.50   | 56.27    | 72.70  | 50.13  |

* REDDIT-M dataset would take a quite long time to load and evaluate, be patient. 

