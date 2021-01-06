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

How to run example files
--------------------------------
In the hgat folder, run

**If you want to test architecture in paper please use `hgat.py`, if you want to test the architecture similar to gat in example. use `train.py`**


```python
python main.py
```

If want to use a GPU, run

```python
python hgat.py --gpu 0
```

If you want to use more Graph Hard Attention Modules

```python
python hgat.py --num-layers <your number>
```

If you want to change the hard attention threshold k

```python
python hgat.py --k <your number>
```

**Similar Arguments can be passed to `train.py`**

Performance
-------------------------
| Models/Datasets | Cora | Citeseer | Pubmed |
| :-------------- | :--: | :------: | -----: |
| GAT in DGL | 81.5% | 70.1% | 77.7% |
| HardGAT similar to GAT | 81.8% | 70.2% |78.0%|
| HardGANet in paper | 77.0% | 64.2% | 77.1% |