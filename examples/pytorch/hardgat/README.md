# HardGAT
## DGL Implementation of h/cGAO paper.

This DGL example implements the GNN model proposed in the paper [HardGraphAttention](https://arxiv.org/abs/1907.04652.pdf). 

HardGANet implementor
----------------------
This example was implemented by [Ericcsr](https://github.com/Ericcsr) during his Internship work at the AWS Shanghai AI Lab.

The graph dataset used in this example 
---------------------------------------
The DGL's built-in CoraGraphDataset. Dataset summary:
- NumNodes: 2708
- NumEdges: 10556
- NumFeats: 1433
- NumClasses: 7
- NumTrainingSamples: 140
- NumValidationSamples: 500
- NumTestSamples: 1000

The DGL's build-in CiteseerGraphDataset. Dataset Summary:

- NumNodes: 3327
- NumEdges: 9228
- NumFeats: 3703
- NumClasses: 6
- NumTrainingSamples: 120
- NumValidationSamples: 500
- NumTestSamples: 1000

The DGL's build-in PubmedGraphDataset. Dataset Summary:

- NumNodes: 19717
- NumEdges: 88651
- NumFeats: 500
- NumClasses: 3
- NumTrainingSamples: 60
- NumValidationSamples: 500
- NumTestSamples: 1000

How to run example files
--------------------------------
In the hgao folder, run

**Please use `train.py`**


```python
python train.py --dataset=cora
```

If want to use a GPU, run

```python
python train.py --gpu 0 --dataset=citeseer
```

If you want to use more Graph Hard Attention Modules

```python
python train.py --num-layers <your number> --dataset=pubmed
```

If you want to change the hard attention threshold k

```python
python train.py --k <your number> --dataset=cora
```

If you want to test with vanillia GAT

```python
python train.py --model <gat/hgat> --dataset=cora
```



Performance
-------------------------
| Models/Datasets | Cora | Citeseer | Pubmed |
| :-------------- | :--: | :------: | -----: |
| GAT in DGL | 81.5% | 70.1% | 77.7% |
| HardGAT | 81.8% | 70.2% |78.0%|

Notice that HardGAT Simply replace GATConv with hGAO mentioned in paper.

