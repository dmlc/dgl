# DGL Implementation of ARMA

This DGL example implements the GNN model proposed in the paper [Graph Neural Networks with convolutional ARMA filters](https://arxiv.org/abs/1901.01343).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.6. For version requirement of packages, see below.

```
dgl
numpy 1.19.5
networkx 2.5
scikit-learn 0.24.1
tqdm 4.56.0
torch 1.7.0
```

### The graph datasets used in this example

###### Node Classification

The DGL's built-in Cora, Pubmed, Citeseer datasets. Dataset summary:

| Dataset | #Nodes | #Edges | #Feats | #Classes | #Train Nodes | #Val Nodes | #Test Nodes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Cora | 2,708 | 10,556 | 1,433 | 7(single label) | 140 | 500 | 1000 |
| Citeseer | 3,327 | 9,228 | 3,703 | 6(single label) | 120 | 500 | 1000 |
| Pubmed | 19,717 | 88,651 | 500 | 3(single label) | 60 | 500 | 1000 |

### Usage

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
--epochs           int     Number of training epochs.          Default is 2000.
--early-stopping   int     Early stopping rounds.              Default is 100.
--lr               float   Adam optimizer learning rate.       Default is 0.01.
--lamb             float   L2 regularization coefficient.      Default is 0.0005.
--hid-dim          int     Hidden layer dimensionalities.      Default is 16.
--num-stacks       int     Number of K.                        Default is 2.
--num-layers       int     Number of T.                        Default is 1.
--dropout          float   Dropout applied at all layers.      Default is 0.75.
```

###### Examples

The following commands learn a neural network and predict on the test set.
Train an ARMA model which follows the original hyperparameters on different datasets.
```bash
# Cora:
python citation.py --gpu 0

# Citeseer:
python citation.py --gpu 0 --dataset Citeseer --num-stacks 3

# Pubmed:
python citation.py --gpu 0 --dataset Pubmed --dropout 0.25 --num-stacks 1
```

### Performance

###### Node Classification

| Dataset | Cora | Citeseer | Pubmed |
| :-: | :-: | :-: | :-: |
| Metrics(Table 1.Node classification accuracy) | 83.4±0.6 | 72.5±0.4 | 78.9±0.3 |
| Metrics(PyG) | 82.3±0.5 | 70.9±1.1 | 78.3±0.8 |
| Metrics(DGL) | 80.9±0.6 | 71.6±0.8 | 75.0±4.2 |