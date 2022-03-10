# DGL Implementations of P-GNN

This DGL example implements the GNN model proposed in the paper [Position-aware Graph Neural Networks](http://proceedings.mlr.press/v97/you19b/you19b.pdf). For the original implementation, see [here](https://github.com/JiaxuanYou/P-GNN).

Contributor: [RecLusIve-F](https://github.com/RecLusIve-F)

## Requirements

The codebase is implemented in Python 3.8. For version requirement of packages, see below.

```
dgl 0.7.2
numpy 1.21.2
torch 1.10.1
networkx 2.6.3
scikit-learn 1.0.2
```

## Instructions for experiments

### Link prediction

```bash
# Communities-T
python main.py --task link

# Communities
python main.py --task link --inductive
```

### Link pair prediction

```bash
# Communities
python main.py --task link_pair --inductive
```

## Performance

### Link prediction (Grid-T and Communities-T refer to the transductive learning setting of Grid and Communities)

|             Dataset              | Communities-T |  Communities  |
| :------------------------------: | :-----------: | :-----------: |
| ROC AUC ( P-GNN-E-2L in Table 1) | 0.988 ± 0.003 | 0.985 ± 0.008 |
|    ROC AUC (DGL: P-GNN-E-2L)     | 0.984 ± 0.010 | 0.991 ± 0.004 |

### Link pair prediction

|             Dataset              | Communities |
| :------------------------------: | :---------: |
| ROC AUC ( P-GNN-E-2L in Table 1) | 1.0 ± 0.001 |
|    ROC AUC (DGL: P-GNN-E-2L)     | 1.0 ± 0.000 |
