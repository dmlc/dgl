# DGL Implementation of DeeperGCN

This DGL example implements the GNN model proposed in the paper [DeeperGCN: All You Need to Train Deeper GCNs](https://arxiv.org/abs/2006.07739). For the original implementation, see [here](https://github.com/lightaime/deep_gcns_torch).

Contributor: [xnuohz](https://github.com/xnuohz)

### Requirements
The codebase is implemented in Python 3.7. For version requirement of packages, see below.

```
dgl 0.6.0.post1
torch 1.7.0
ogb 1.3.0
```

### The graph datasets used in this example

Open Graph Benchmark(OGB). Dataset summary:

###### Graph Property Prediction

|   Dataset   | #Graphs | #Node Feats | #Edge Feats | Metric  |
| :---------: | :-----: | :---------: | :---------: | :-----: |
| ogbg-molhiv | 41,127  |      9      |      3      | ROC-AUC |

### Usage

Train a model which follows the original hyperparameters on different datasets.
```bash
# ogbg-molhiv
python main.py --gpu 0 --learn-beta
```

### Performance

* Table 6: Numbers associated with "Table 6" are the ones from table 6 in the paper.
* Author: Numbers associated with "Author" are the ones we got by running the original code.
* DGL: Numbers associated with "DGL" are the ones we got by running the DGL example.

|     Dataset      | ogbg-molhiv |
| :--------------: | :---------: |
| Results(Table 6) |    0.786    |
| Results(Author)  |    0.781    |
|   Results(DGL)   |    0.778    |

### Speed

|     Dataset     | ogbg-molhiv |
| :-------------: | :---------: |
| Results(Author) |   11.833    |
|  Results(DGL)   |    8.965    |
