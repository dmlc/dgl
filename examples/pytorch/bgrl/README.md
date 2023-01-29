# DGL Implementation of BGRL

This DGL example implements the GNN experiment proposed in the paper [Large-Scale Representation Learning on Graphs via Bootstrapping](https://arxiv.org/abs/2102.06514). For the original implementation, see [here](https://github.com/nerdslab/bgrl).

Contributor: [RecLusIve-F](https://github.com/RecLusIve-F)

### Requirements

The codebase is implemented in Python 3.8. For version requirement of packages, see below.

```
dgl 0.8.3
numpy 1.21.2
torch 1.10.2
scikit-learn 1.0.2
```

### Dataset
Dataset summary:

|     Dataset      |     Task     | Nodes  |  Edges  | Features |     Classes     |
|:----------------:|:------------:|:------:|:-------:|:--------:|:---------------:|
|      WikiCS      | Transductive | 11,701 | 216,123 |   300    |       10        |
| Amazon Computers | Transductive | 13,752 | 245,861 |   767    |       10        |
|  Amazon Photos   | Transductive | 7,650  | 119,081 |   745    |        8        |
|   Coauthor CS    | Transductive | 18,333 | 81,894  |  6,805   |       15        |
| Coauthor Physics | Transductive | 34,493 | 247,962 |  8,415   |        5        |
|  PPI(24 graphs)  |  Inductive   | 56,944 | 818,716 |    50    | 121(multilabel) |

### Usage

##### Dataset options
```
--dataset                     str         The graph dataset name.                         Default is 'amazon_photos'.
```

##### Model options
```
--graph_encoder_layer         list        Convolutional layer hidden sizes.               Default is [256, 128].
--predictor_hidden_size       int         Hidden size of predictor.                       Default is 512.
```

##### Training options
```
--epochs                      int         The number of training epochs.                  Default is 10000.
--lr                          float       The learning rate.                              Default is 0.00001.
--weight_decay                float       The weight decay.                               Default is 0.00001.
--mm                          float       The momentum for moving average.                Default is 0.99.
--lr_warmup_epochs            int         Warmup period for learning rate scheduling.     Default is 1000.    
--weights_dir                 str         Where to save the weights.                      Default is '../weights'.
```

##### Augmentation options
```
--drop_edge_p                 float      Probability of edge dropout.                     Default is [0., 0.].
--feat_mask_p                 float      Probability of node feature masking.             Default is [0., 0.].
```

##### Evaluation options
```
--eval_epochs                 int        Evaluate every eval_epochs.                      Default is 250.
--num_eval_splits             int        Number of evaluation splits.                     Default is 20.
--data_seed                   int        Data split seed for evaluation.                  Default is 1.
```

### Instructions for experiments

##### Transductive task
```
# Coauthor CS
python main.py --dataset coauthor_cs --graph_encoder_layer 512 256 --drop_edge_p 0.3 0.2 --feat_mask_p 0.3 0.4

# Coauthor Physics
python main.py --dataset coauthor_physics --graph_encoder_layer 256 128 --drop_edge_p 0.4 0.1 --feat_mask_p 0.1 0.4

# WikiCS
python main.py --dataset wiki_cs --graph_encoder_layer 512 256 --drop_edge_p 0.2 0.3 --feat_mask_p 0.2 0.1 --lr 5e-4

# Amazon Photos
python main.py --dataset amazon_photos --graph_encoder_layer 256 128 --drop_edge_p 0.4 0.1 --feat_mask_p 0.1 0.2 --lr 1e-4

# Amazon Computers
python main.py --dataset amazon_computers --graph_encoder_layer 256 128 --drop_edge_p 0.5 0.4 --feat_mask_p 0.2 0.1 --lr 5e-4
```

##### Inductive task
```
# PPI
python main.py --dataset ppi --graph_encoder_layer 512 512 --drop_edge_p 0.3 0.25 --feat_mask_p 0.25 0. --lr 5e-3
```

### Performance

##### Transductive Task
|        Dataset         |    WikiCS    |  Am. Comp.   |  Am. Photos  |    Co. CS    |   Co. Phy    |
|:----------------------:|:------------:|:------------:|:------------:|:------------:|:------------:|
|   Accuracy Reported    | 79.98 ± 0.10 | 90.34 ± 0.19 | 93.17 ± 0.30 | 93.31 ± 0.13 | 95.73 ± 0.05 |
| Accuracy Official Code |    79.94     |    90.62     |    93.45     |    93.42     |    95.74     |
|      Accuracy DGL      |    80.00     |    90.64     |    93.34     |    93.76     |    95.79     |

##### Inductive Task
|        Dataset         |     PPI      |
|:----------------------:|:------------:|
|   Micro-F1 Reported    | 69.41 ± 0.15 |
| Accuracy Official Code |    68.83     |
|      Micro-F1 DGL      |    68.65     |


##### Accuracy reported is over 20 random dataset splits and model initializations. Micro-F1 reported is over 20 random model initializations.

##### Accuracy official code and Accuracy DGL is only over 1 random dataset splits and model initialization. Micro-F1 official code and Micro-F1 DGL is only over 1 random model initialization.