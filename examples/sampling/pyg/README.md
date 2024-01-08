##  Overview

This project demonstrates the training and evaluation of a GraphSAGE model for node classification on large graphs. The example utilizes GraphBolt for efficient data handling and PyG for the GNN training.


# Node classification on graph

This example aims to demonstrate how to run node classification task on heterogeneous graph with **GraphBolt**. 

##  Model

The model is a three-layer GraphSAGE network implemented using PyTorch Geometric's SAGEConv layers.


## Training is performed with the following settings:

Mini-batch size: 1024 Neighbor sampling: [10, 10, 10] Optimizer: Ada Learning Rate: 0.01 Weight Decay: 5e-4 Loss Function: CrossEntropyLoss Evaluation is done separately for validation and test datasets. The model's performance is measured in terms of accuracy using the torchmetrics.functional library.


## Run on `ogbn-arxiv` dataset


### Sample on CPU and train/infer on CPU

python3 node_classification.py


### Accuracies

Final performance: 
All runs:
Highest Train: 62.26
Highest Valid: 59.89
Final Train: 62.26
Final Test: 52.78





