# Graph Random Neural Network(GRAND)

This DGL example implements the GNN model proposed in the paper [Graph Random Neural Network for Semi-Supervised Learning on Graphs]( https://arxiv.org/abs/2005.11079).

Author's code: https://github.com/THUDM/GRAND

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) when he was an applied scientist intern at AWS Shanghai AI Lab.

## Dependencies
- Python 3.7
- PyTorch 1.7.1
- dgl 0.5.3

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
--dataname          str     The graph dataset name.             Default is 'cora'.
```

###### GPU options
```
--gpu              int     GPU index.                          Default is -1, using CPU.
```

###### Model options
```
--epochs           int     Number of training epochs.             Default is 2000.
--early_stopping   int     Early stopping patience rounds.        Default is 200.
--lr               float   Adam optimizer learning rate.          Default is 0.01.
--weight_decay     float   L2 regularization coefficient.         Default is 5e-4.
--dropnode_rate    float   Dropnode rate (1 - keep probability).  Default is 0.5.
--input_droprate   float   Dropout rate of input layer.           Default is 0.5.
--hidden_droprate  float   Dropout rate of hidden layer.          Default is 0.5.
--hid_dim          int     Hidden layer dimensionalities.         Default is 32.
--order            int     Propagation step.                      Default is 8.
--sample           int     Sampling times of dropnode.            Default is 4.
--tem              float   Sharpening temperaturer.               Default is 0.5.
--lam              float   Coefficient of Consistency reg         Default is 1.0.
--use_bn           bool    Using batch normalization.             Default is False
```

## Examples

Train a model which follows the original hyperparameters on different datasets.
```bash
# Cora:
python main.py --dataname cora --gpu 0 --lam 1.0 --tem 0.5 --order 8 --sample 4 --input_droprate 0.5 --hidden_droprate 0.5 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 100 --lr 1e-2  --epochs 2000
# Citeseer:
python main.py --dataname citeseer --gpu 0 --lam 0.7 --tem 0.3 --order 2 --sample 2 --input_droprate 0.0 --hidden_droprate 0.2 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 100 --lr 1e-2  --epochs 2000
# Pubmed:
python main.py --dataname pubmed --gpu 0 --lam 1.0 --tem 0.2 --order 5 --sample 4 --input_droprate 0.6 --hidden_droprate 0.8 --dropnode_rate 0.5 --hid_dim 32 --early_stopping 200 --lr 0.2 --epochs 2000 --use_bn
```

### Performance

The hyperparameter setting in our implementation is identical to that reported in the paper.

| Dataset | Cora | Citeseer | Pubmed |
| :-: | :-: | :-: | :-: |
| Accuracy Reported(100 runs) | **85.4(±0.4)** | **75.4(±0.4)** | 82.7(±0.6) |
| Accuracy DGL(20 runs) | 85.33(±0.41) | 75.36(±0.36) | **82.90(±0.66)** |



