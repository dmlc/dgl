# DGL Implementation of the GNN-FiLM Model

This DGL example implements the GNN model proposed in the paper [GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation](https://arxiv.org/pdf/1906.12192.pdf). 
The author's codes of implementation is in [here](https://github.com/Microsoft/tf-gnn-samples)


Example implementor
----------------------
This example was implemented by [Kounianhua Du](https://github.com/KounianhuaDu) during her Software Dev Engineer Intern work at the AWS Shanghai AI Lab.


Dependencies
----------------------
- numpy 1.19.4
- scikit-learn 0.22.1
- pytorch 1.4.0
- dgl 0.5.3


The graph dataset used in this example 
---------------------------------------
The DGL's built-in PPIDataset. This is a Protein-Protein Interaction dataset for inductive node classification. The PPIDataset is a toy Protein-Protein Interaction network dataset. The dataset contains 24 graphs. The average number of nodes per graph is 2372. Each node has 50 features and 121 labels. There are 20 graphs for training, 2 for validation, and 2 for testing.

NOTE: Following the paper, in addition to the dataset-provided untyped edges, a fresh "self-loop" edge type is added.

Statistics:
- Train examples: 20
- Valid examples: 2
- Test examples: 2
- AvgNodesPerGraph: 2372
- NumFeats: 50
- NumLabels: 121


How to run example files
--------------------------------
In the GNNFiLM folder, run

```bash
python main.py 
```

If want to use a GPU, run

```bash
python main.py --gpu ${your_device_id_here}
```


Performance
-------------------------

NOTE: We do not perform grid search or finetune here, so there is a gap between the performance reported in the original paper and this example. Below results, mean(standard deviation), were computed over ten runs.

**GNN-FiLM results on PPI task**
| Model         | Paper (tensorflow)               | ours (dgl)                  |
| ------------- | -------------------------------- | --------------------------- |
| Avg. Micro-F1 | 0.992 (0.000)                    | 0.983 (0.001)               |
