# DAGNN

This DGL example implements the GNN model proposed in the paper [Towards Deeper Graph Neural Networks](https://arxiv.org/abs/2007.09296).

Paper link: https://arxiv.org/abs/2007.09296

Author's code: https://github.com/divelab/DeeperGNN

Contributor: Liu Tang ([@lt610](https://github.com/lt610))

## Dependecies
- Python 3.6.10
- PyTorch 1.4.0
- numpy 1.18.1
- dgl 0.5.3
- tqdm 4.44.1

## Dataset

The DGL's built-in Cora, Pubmed and Citeseer datasets. Dataset summary:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Citeseer | 3,327 | 9,228 | 3,703 | 6 | 120 | 500 | 1000 |
| Cora | 2,708 | 10,556 | 1,433 | 7 | 140 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3 | 60 | 500 | 1000 |

## Arguments

###### Dataset options
```
--dataset          str     The graph dataset name.             Default is 'Cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--runs             int     Number of training runs.               Default is 1
--epochs           int     Number of training epochs.             Default is 1500.
--early-stopping   int     Early stopping patience rounds.        Default is 100.
--lr               float   Adam optimizer learning rate.          Default is 0.01.
--lamb             float   L2 regularization coefficient.         Default is 5e-3.
--k                int     Number of propagation layers.          Default is 10.
--hid-dim          int     Hidden layer dimensionalities.         Default is 64.
--dropout          float   Dropout rate                           Default is 0.8
```

## Examples

Train a model which follows the original hyperparameters on different datasets.
```bash
# Cora:
python main.py --dataset Cora --gpu 0 --runs 100 --lamb 0.005 --k 12
# Citeseer:
python main.py --dataset Citeseer --gpu 0 --runs 100 --lamb 0.02 --k 16
# Pubmed:
python main.py --dataset Pubmed --gpu 0 --runs 100 --lamb 0.005 --k 20
```
### Performance

#### On Cora, Citeseer and Pubmed
| Dataset | Cora | Citeseer | Pubmed |
| :-: | :-: | :-: | :-: |
| Accuracy Reported(100 runs) | 84.4 ± 0.5 | 73.3 ± 0.6 | 80.5 ± 0.5 |
| Accuracy DGL(100 runs) | 84.3 ± 0.5 | 73.1 ± 0.9 | 80.5 ± 0.4 |
