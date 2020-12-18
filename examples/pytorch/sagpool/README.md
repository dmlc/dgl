# DGL Implementation of the SAGPool Paper

This DGL example implements the GNN model proposed in the paper [Self Attention Graph Pooling](https://arxiv.org/pdf/1904.08082.pdf). 
The author's codes of implementation is in [here](https://github.com/inyeoplee77/SAGPool)


The graph dataset used in this example 
---------------------------------------
The DGL's built-in LegacyTUDataset. This is a serial of graph kernel datasets for graph classification. We use 'DD', 'PROTEINS', 'NCI1', 'NCI109' and 'Mutagenicity' in this SAGPool implementation. All these datasets are randomly splited to train, validation and test set with ratio 0.8, 0.1 and 0.1.

NOTE: Since there is no data attributes in some of these datasets, we use node_id (in one-hot vector whose length is the max number of nodes across all graphs) as the node feature. Also note that the node_id in some datasets is not unique (e.g. a graph may has two nodes with the same id).

DD
- NumGraphs: 1178
- AvgNodesPerGraph: 284.32
- AvgEdgesPerGraph: 715.66
- NumFeats: 89
- NumClasses: 2

PROTEINS
- NumGraphs: 1113
- AvgNodesPerGraph: 39.06
- AvgEdgesPerGraph: 72.82
- NumFeats: 1
- NumClasses: 2

NCI1
- NumGraphs: 4110
- AvgNodesPerGraph: 29.87
- AvgEdgesPerGraph: 32.30
- NumFeats: 37
- NumClasses: 2

NCI109
- NumGraphs: 4127
- AvgNodesPerGraph: 29.68
- AvgEdgesPerGraph: 32.13
- NumFeats: 38
- NumClasses: 2

Mutagenicity
- NumGraphs: 4337
- AvgNodesPerGraph: 30.32
- AvgEdgesPerGraph: 30.77
- NumFeats: 14
- NumClasses: 2


How to run example files
--------------------------------
The valid dataset names (you can find a full list [here](https://chrsmrrs.github.io/datasets/docs/datasets/)):
- 'DD' for D&D
- 'PROTEINS' for PROTEINS
- 'NCI1' for NCI1
- 'NCI109' for NCI109
- 'Mutagenicity' for Mutagenicity

In the sagpool folder, run

```bash
python main.py --dataset ${your_dataset_name_here}
```

If want to use a GPU, run

```bash
python main.py --device ${your_device_id_here} --dataset ${your_dataset_name_here}
```

If your want to perform a grid search, modify parameter settings in `grid_search_config.json` and run
```bash
python grid_search.py --device ${your_device_id_here} --num_trials ${num_of_trials_here}
```

Performance
-------------------------

NOTE: We do not perform grid search or finetune here, so there may be a gap between results in paper and our results. Also, we only perform 10 trials for each experiment, which is different from 200 trials per experiment in the paper.

**The global architecture result**
| Dataset       | paper result (global)            | ours (global)               |
| ------------- | -------------------------------- | --------------------------- |
| D&D           | 76.19 (0.94)                     | 74.12 (2.61)                |
| PROTEINS      | 70.04 (1.47)                     | 69.46 (3.52)                |
| NCI1          | 74.18 (1.20)                     | 72.72 (1.77)                |
| NCI109        | 74.06 (0.78)                     | 71.21 (1.68)                |
| Mutagenicity  | N/A                              | 74.51 (1.97)                |

**The hierarchical architecture result**
| Dataset       | paper result (hierarchical)      | ours (hierarchical)         |
| ------------- | -------------------------------- | --------------------------- |
| D&D           | 76.45 (0.97)                     | 76.47 (4.45)                |
| PROTEINS      | 71.86 (0.97)                     | 72.41 (1.89)                |
| NCI1          | 67.45 (1.11)                     | 72.34 (1.69)                |
| NCI109        | 67.86 (1.41)                     | 69.73 (1.41)                |
| Mutagenicity  | N/A                              | 74.71 (2.74)                |
