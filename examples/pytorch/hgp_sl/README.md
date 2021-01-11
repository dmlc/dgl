# DGL Implementation of the HGP-SL Paper

This DGL example implements the GNN model proposed in the paper [Hierarchical Graph Pooling with Structure Learning](https://arxiv.org/pdf/1911.05954.pdf). 
The author's codes of implementation is in [here](https://github.com/cszhangzhen/HGP-SL)


Example implementor
----------------------
This example was implemented by [Tianqi Zhang](https://github.com/lygztq) during his Applied Scientist Intern work at the AWS Shanghai AI Lab.


The graph dataset used in this example 
---------------------------------------
The DGL's built-in LegacyTUDataset. This is a serial of graph kernel datasets for graph classification. We use 'DD', 'PROTEINS', 'NCI1', 'NCI109', 'Mutagenicity' and 'ENZYMES' in this HGP-SL implementation. All these datasets are randomly splited to train, validation and test set with ratio 0.8, 0.1 and 0.1.

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

ENZYMES
- NumGraphs: 600
- AvgNodesPerGraph: 32.63
- AvgEdgesPerGraph: 62.14
- NumFeats: 18
- NumClasses: 6

How to run example files
--------------------------------
In the HGP-SL-DGL folder, run

```bash
python main.py --dataset ${your_dataset_name_here}
```

If want to use a GPU, run

```bash
python main.py --device ${your_device_id_here} --dataset ${your_dataset_name_here}
```

Performance
-------------------------

**Hyper-parameters**

This part is directly from [author's implementation](https://github.com/cszhangzhen/HGP-SL)

| Datasets      | lr        | weight_decay   | batch_size      | pool_ratio     | dropout  | net_layers |
| ------------- | --------- | -------------- | --------------- | -------------- | -------- | ---------- |
| PROTEINS      | 0.001     | 0.001          | 512             | 0.5            | 0.0      | 3          | 
| Mutagenicity  | 0.001     | 0.001          | 512             | 0.8            | 0.0      | 3          |
| NCI109        | 0.001     | 0.001          | 512             | 0.8            | 0.0      | 3          |
| NCI1          | 0.001     | 0.001          | 512             | 0.8            | 0.0      | 3          |
| DD            | 0.0001    | 0.001          | 64              | 0.3            | 0.5      | 2          |
| ENZYMES       | 0.001     | 0.001          | 128             | 0.8            | 0.0      | 2          |


**Accuracy**

**NOTE**: We find that there is a gap between accuracy obtained via author's code and the one reported in the [paper]((https://arxiv.org/pdf/1911.05954.pdf)). An issue has been proposed in the author's repo (see [here](https://github.com/cszhangzhen/HGP-SL/issues/8)).

|                            | Mutagenicity | NCI109      | NCI1        | DD          |
| -------------------------- | ------------ | ----------- | ----------- | ----------- |
| Reported in Paper          | 82.15(0.58)  | 80.67(1.16) | 78.45(0.77) | 80.96(1.26) |
| Author's Code (full graph) | 78.44(2.10)  | 74.44(2.05) | 77.37(2.09) | OOM         |
| Author's Code (sample)     | 79.68(1.68)  | 73.86(1.72) | 76.29(2.14) | 75.46(3.86) |
| DGL (full graph)           | 79.52(2.21)  | 74.86(1.99) | 74.62(2.22) | OOM         |
| DGL (sample)               | 79.15(1.62)  | 75.39(1.86) | 73.77(2.04) | 76.47(2.14) |


**Speed**

Device: Tesla V100-SXM2 16GB

In seconds

|                               | DD(batchsize=64), large graph | Mutagenicity(batchsize=512), small graph |
| ----------------------------- | ----------------------------- | ---------------------------------------- |
| Author's code (sample)        | 9.96                          | 12.91                                    |
| Author's code (full graph)    | OOM                           | 13.03                                    |
| DGL (sample)                  | 9.50                          | 3.59                                     |
| DGL (full graph)              | OOM                           | 3.56                                     |
