# DGL Implementation of InfoGraph
This DGL example implements the model proposed in the paper [InfoGraph: Unsupervised and Semi-supervised Graph-Level Representation Learning via Mutual Information Maximization](https://arxiv.org/abs/1908.01000).

Author's code: https://github.com/fanyun-sun/InfoGraph

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) when he was an applied scientist intern at AWS Shanghai AI Lab.

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0

## Datasets

##### Unsupervised Graph Classification Dataset:

 'MUTAG', 'PTC', 'IMDBBINARY'(IMDB-B), 'IMDBMULTI'(IMDB-M), 'REDDITBINARY'(RDT-B), 'REDDITMULTI5K'(RDT-M5K) of dgl.data.GINDataset.

| Dataset         | MUTAG | PTC   | RDT-B  | RDT-M5K | IMDB-B | IMDB-M |
| --------------- | ----- | ----- | ------ | ------- | ------ | ------ |
| # Graphs        | 188   | 344   | 2000   | 4999    | 1000   | 1500   |
| # Classes       | 2     | 2     | 2      | 5       | 2      | 3      |
| Avg. Graph Size | 17.93 | 14.29 | 429.63 | 508.52  | 19.77  | 13.00  |

**Semi-supervised Graph Regression Dataset:**

QM9 dataset for graph property prediction (regression)

| Dataset | # Graphs | # Regression Tasks |
| ------- | -------- | ------------------ |
| QM9     | 130,831  | 12                 |

The 12 tasks are:

| Keys  | Description                                |
| ----- | :----------------------------------------- |
| mu    | Dipole moment                              |
| alpha | Isotropic polarizability                   |
| homo  | Highest occupied molecular orbital energ   |
| lumo  | Lowest unoccupied molecular orbital energy |
| gap   | Gap between 'homo' and 'lumo'              |
| r2    | Electronic spatial extent                  |
| zpve  | Zero point vibrational energy              |
| U0    | Internal energy at 0K                      |
| U     | Internal energy at 298.15K                 |
| H     | Enthalpy at 298.15K                        |
| G     | Free energy at 298.15K                     |
| Cv    | Heat capavity at 298.15K                   |

## Arguments

##### 	Unsupervised Graph Classification:

###### Dataset options

```
--dataname        str      The graph dataset name.                Default is 'MUTAG'.
```

###### GPU options

```
--gpu              int     GPU index.                             Default is -1, using CPU.
```

###### Training options

```
--epochs           int     Number of training periods.            Default is 20.
--batch_size       int     Size of a training batch.              Default is 128.
--lr               float   Adam optimizer learning rate.          Default is 0.01.
--log_interval     int     Interval bettwen two evaluations.	  Default is 1.
```

###### Model options

```
--n_layers         int     Number of GIN layers.                  Default is 3.
--hid_dim          int     Dimension of hidden layers.            Default is 32.
```

##### 	Semi-supervised Graph Regression:

###### Dataset options

```
 --target          str     The regression Task.                   Default is 'mu'.
 --train_num       int     Number of supervised examples.         Default is 5000.
```

###### GPU options

```
--gpu              int     GPU index.                             Default is -1, using CPU.
```

###### Training options

```
--epochs           int     Number of training periods.            Default is 200.
--batch_size       int     Size of a training batch.              Default is 20.
--val_batch_size   int     Size of a validation batch.            Default is 100.
--lr               float   Adam optimizer learning rate.          Default is 0.001.
```

###### Model options

```
--hid_dim          int     Dimension of hidden layers.            Default is 64.
--reg              int     Regularization weight.                 Default is 0.001.
```

## How to run examples

Training and testing unsupervised model on MUTAG.

 (As graphs in these datasets are quite small and sparse, moving graphs from cpu to gpu would take a longer time than training, we recommend using **cpu** for these datasets).

```bash
# MUTAG:
python unsupervised.py --dataname MUTAG --n_layers 4 --hid_dim 32
```

Replace 'MUTAG' with dataname in ['MUTAG', 'PTC', 'IMDBBINARY', 'IMDBMULTI', 'REDDITBINARY', 'REDDITMULTI5K'] if you'd like to try other datasets.

Training and testing semi-supervised model on QM9 for graph property 'mu' with gpu.

```bash
# QM9:
python semisupervised.py --gpu 0 --target mu
```

Replace 'mu' with other target names above.

## 	Performance

The hyperparameter setting in our implementation is identical to that reported in the paper.

##### Unsupervised Graph Classification:

|      Dataset      | MUTAG |  PTC  | RDT-B | RDT-M5K | IMDB-B | IMDB-M |
| :---------------: | :---: | :---: | :---: | ------- | ------ | ------ |
| Accuracy Reported | 89.01 | 61.65 | 82.50 | 53.46   | 73.03  | 49.69  |
|        DGL        | 89.88 | 63.54 | 88.50 | 56.27   | 72.70  | 50.13  |

* REDDIT-M dataset would take a quite long time to load and evaluate. 

##### Semisupervised Graph Regression on QM9:

Here we only provide the results of 'mu', 'alpha', 'homo'.



|      Target       |   mu   | alpha  |  homo  |
| :---------------: | :----: | :----: | :----: |
|   MAE Reported    | 0.3169 | 0.5444 | 0.0060 |
| The authors' code | 0.2411 | 0.5192 | 0.1560 |
|        DGL        | 0.2355 | 0.5483 | 0.1581 |

* The source of QM9 Dataset has changed so there's a gap between the MAE reported in the paper and that we reprodcued.
* See this [issue](https://github.com/fanyun-sun/InfoGraph/issues/8) for authors' response. 
