# DGL Implementation of GRACE
This DGL example implements the model proposed in the paper [Deep Graph Contrastive Representation Learning](https://arxiv.org/abs/2006.04131).

Author's code: https://github.com/CRIPAC-DIG/GRACE

## Example Implementor

This example was implemented by [Hengrui Zhang](https://github.com/hengruizhang98) when he was an applied scientist intern at AWS Shanghai AI Lab.

## Dependencies

- Python 3.7
- PyTorch 1.7.1
- dgl 0.6.0
- scikit-learn 0.22.1

## Datasets

##### Unsupervised Node Classification Datasets:

'Cora', 'Citeseer' and 'Pubmed'

| Dataset  | # Nodes | # Edges | # Classes |
| -------- | ------- | ------- | --------- |
| Cora     | 2,708   | 10,556  | 7         |
| Citeseer | 3,327   | 9,228   | 6         |
| Pubmed   | 19,717  | 88,651  | 3         |


## Arguments

```
--dataname         str     The graph dataset name.                Default is 'cora'.
--gpu              int     GPU index.                             Default is 0.
--split            int     Dataset spliting method.               Default is 'random'.
--epochs           int     Number of training periods.            Default is 500.
--lr               float   Learning rate.                         Default is 0.001.
--wd               float   Weight decay.                          Default is 1e-5.
--temp             float   Temperature.                           Default is 1.0.
--act_fn           str     Activation function.                   Default is relu.
--hid_dim          int     Hidden dimension.                      Default is 256.
--out_dim          int     Output dimension.                      Default is 256.
--num_layers       int     Number of GNN layers.                  Default is 2.
--der1             float   Drop edge ratio 1.                     Default is 0.2. 
--der2             float   Drop edge ratio 2.                     Default is 0.2. 
--dfr1             float   Drop feature ratio 1.                  Default is 0.2. 
--dfr2             float   Drop feature ratio 2.                  Default is 0.2. 
```

## How to run examples

In the paper(as well as authors' repo), the training set and testing set are split randomly with 1:9 ratio. In order to fairly compare it with other methods with the public split (20 training nodes each class), in this repo we also provide its results using the public split (with fine-tuned hyper-parameters). To run the examples, follow the following instructions.

```python
# Cora with random split
python main.py --dataname cora --epochs 200 --lr 5e-4 --wd 1e-5 --hid_dim 128 --out_dim 128 --act_fn relu --der1 0.2 --der2 0.4 --dfr1 0.3 --dfr2 0.4 --temp 0.4

# Cora with public split
python main.py --dataname cora --split public --epochs 400 --lr 5e-4 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.3 --der2 0.4 --dfr1 0.3 --dfr2 0.4 --temp 0.4

# Citeseer with random split
python main.py --dataname citeseer --epochs 200 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn prelu --der1 0.2 --der2 0.0 --dfr1 0.3 --dfr2 0.2 --temp 0.9

# Citeseer with public split
python main.py --dataname citeseer --split public --epochs 100 --lr 1e-3 --wd 1e-5 --hid_dim 512 --out_dim 512 --act_fn prelu --der1 0.3 --der2 0.3 --dfr1 0.3 --dfr2 0.3 --temp 0.4

# Pubmed with random split
python main.py --dataname pubmed --epochs 1500 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.4 --der2 0.1 --dfr1 0.0 --dfr2 0.2 --temp 0.7

# Pubmed with public split
python main.py --dataname pubmed --split public --epochs 1500 --lr 1e-3 --wd 1e-5 --hid_dim 256 --out_dim 256 --act_fn relu --der1 0.4 --der2 0.1 --dfr1 0.0 --dfr2 0.2 --temp 0.7
```

## 	Performance

For random split, we use the hyper-parameters as stated in the paper. For public split,  we find the given hyper-parameters lead to poor performance, so we select the hyperparameters via a small grid search.

Random split (Train/Test = 1:9)

|      Dataset      | Cora | Citeseer | Pubmed |
| :---------------: | :--: | :------: | :----: |
| Accuracy Reported | 83.3 |   72.1   |  86.7  |
|   Author's Code   | 83.1 |   71.0   |  86.3  |
|        DGL        | 83.4 |   71.4   |  86.1  |

Public split

|    Dataset    | Cora | Citeseer | Pubmed |
| :-----------: | :--: | :------: | :----: |
| Author's Code | 81.9 |   71.2   |  80.6  |
|      DGL      | 82.2 |   71.4   |  80.2  |

