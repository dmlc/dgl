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


### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.4xlarge**, an Intel(R) Xeon(R) Platinum 8259CL CPU with 62 GB of RAM, NVIDIA Tesla T4 GPU, boasting a total memory of 15360 MB. During observation. CPU RAM usage is the peak value recorded by `memory_usage` command.



| Dataset Size           | CPU RAM Usage | GPU RAM Usage | Num of GPUs | Total(Training) Time     | 
| ---------------------- | ------------- | ------------- | ----------- | ------------------------ |
| ~80.7MB(Pure PyG)      | ~609  MB      | ~0MB          | 0           | ~125s                    | 
| ~80.7MB(Pure PyG)      | ~849MB        | ~260MB        | 1           | ~39s                     |
| ~80.7MB                | ~645  MB      | ~0MB          | 0           | ~48s                     | 
| ~80.7MB                | ~765MB        | ~156MB        | 1           | ~14s                     |



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


### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.4xlarge**, an Intel(R) Xeon(R) Platinum 8259CL CPU with 62 GB of RAM, NVIDIA Tesla T4 GPU, boasting a total memory of 15360 MB. During observation. CPU RAM usage is the peak value recorded by `memory_usage` command.

| Dataset Size          | CPU RAM Usage | GPU RAM Usage | Num of GPUs | Total(Training) Time     | 
| --------------------  | ------------- | ------------- | ----------- | ------------------------ |
| ~1.4GB(Pure PyG)      | ~10977MB      | ~0MB          | 0           | ~8726s                   |
| ~1.4GB(Pure PyG)      | ~14774MB      | ~5166MB       | 1           | ~1141s                   |
| ~1.4GB                | ~2018MB       | ~0MB          | 0           | ~5360s                   |
| ~1.4GB                | ~2451MB       | ~2343MB       | 1           | ~466s                    |



