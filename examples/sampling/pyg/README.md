##  Overview

This project demonstrates the training and evaluation of a GraphSAGE model for node classification on large graphs. The example utilizes GraphBolt for efficient data handling and PyG for the GNN training.


# Node classification on graph

This example aims to demonstrate how to run node classification task on heterogeneous graph with **GraphBolt**. 

##  Model

The model is a three-layer GraphSAGE network implemented using PyTorch Geometric's SAGEConv layers.


## Default Run on `ogbn-arxiv` dataset

```
python node_classification.py
```




## Accuracies
```
Final performance(for ogbn-arxiv): 
All runs:
Highest Train: 62.26
Highest Valid: 59.89
Final Train: 62.26
Final Test: 52.78
```



## Run on `ogbn-products` dataset

### Sample on CPU and train/infer on CPU

```
python node_classification.py --dataset ogbn-products
```

## Accuracies
```
Final performance(for ogbn-products): 
All runs:
Highest Train: 90.79
Highest Valid: 89.86
Final Train: 90.79
Final Test: 75.24
```





