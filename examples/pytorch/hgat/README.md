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
In the MVP4ModelExample folder, run

```python
python main.py
```

If want to use a GPU, run

```python
python main.py --gpu 0
```

If you want to use more Graph Hard Attention Modules

```python
python main.py --num_module <your number>
```

If you want to change the hard attention threshold k

```python
python main.py --k <your number>
```

Performance
-------------------------
**TODO: Debug the implementation**

**TODO: Compare Cora Performance**

**TODO: Compare Performance in other Node classification ds**

**TODO: Implement Graph Classification Pipeline**
