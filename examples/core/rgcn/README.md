# Node classification on heterogeneous graph with RGCN

This example aims to demonstrate how to run node classification task on heterogeneous graph with **DGL**. Models are not tuned to achieve the best accuracy yet.

## Run on `ogbn-mag` dataset
In the preprocess stage, reverse edges are added and duplicate edges are removed. Feature data of `author` and `institution` node types are generated dynamically with embedding layer.

### Sample on CPU and train/infer on CPU
```
python3 hetero_rgcn.py --dataset ogbn-mag
```

### Sample on CPU and train/infer on GPU
```
python3 hetero_rgcn.py --dataset ogbn-mag --num_gpus 1
```

### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.metal**, 384GB RAM, 96 vCPUs(Cascade Lake P-8259L), 8 NVIDIA T4 GPUs(16GB RAM). CPU RAM usage is the peak value of `used` field of `free` command which is a bit rough. Please refer to `RSS`/`USS`/`PSS` which are more accurate. GPU RAM usage is the peak value recorded by `nvidia-smi` command.

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ----------- | ---------- | --------- | ---------------------------    |
| ~1.1GB       | ~5GB          | 0           |  0GB       | ~4min03s(615it, 2.53it/s)   | ~0min22s(154it, 6.86it/s) + ~0min2s(16it, 6.92it/s) + ~0min1s(11it, 7.34it/s)   |
| ~1.1GB       | ~3GB          | 1           |  4.4GB     | ~1min20s(615it, 7.65it/s)   | ~0min14s(154it, 10.79it/s) + ~0min1s(16it, 10.07it/s) + ~0min1s(11it, 10.42it/s)   |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 83.22 ± 0.00
Highest Valid: 48.25 ± 0.20
  Final Train: 68.45 ± 9.81
   Final Test: 47.51 ± 0.19
```

## Run on `ogb-lsc-mag240m` dataset
In the preprocess stage, reverse edges are added and duplicate edges are removed. What's more, feature data are generated in advance for `author` and `institution` node types via message passing. Since such preprocessing will usually take a long time, we also offer the above files for download:

* [`paper-feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/paper-feat.npy)
* [`author-feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/author-feat.npy)
* [`inst-feat.npy`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/inst-feat.npy)
* [`hetero-graph.dgl`](https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/hetero-graph.dgl)

### Sample on CPU and train/infer on CPU
```
python3 hetero_rgcn.py --dataset ogb-lsc-mag240m
```

### Sample on CPU and train/infer on GPU
```
python3 hetero_rgcn.py --dataset ogb-lsc-mag240m --num_gpus 1
```

### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.metal**, 384GB RAM, 96 vCPUs(Cascade Lake P-8259L), 8 NVIDIA T4 GPUs(16GB RAM). CPU RAM usage is the peak value of `used` field of `free` command which is a bit rough. Please refer to `RSS`/`USS`/`PSS` which are more accurate. GPU RAM usage is the peak value recorded by `nvidia-smi` command.

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ----------- | ---------- | --------- | ---------------------------    |
| ~404GB       | ~60GB       | 0           |  0GB       | ~4min12s(1087it, 4.31it/s)  | ~2min40s(272it, 1.70it/s) + ~0min25s(34it, 1.35it/s) + ~0min15s(22it, 1.43it/s)   |
| ~404GB       | ~60GB       | 1           |  7GB       | ~2min46s(1087it, 6.52it/s)  | ~1min49s(272it, 2.48it/s) + ~0min17s(34it, 1.76it/s) + ~0min12s(22it, 1.81it/s)  |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 54.85 ± 1.02
Highest Valid: 52.29 ± 0.50
  Final Train: 54.78 ± 1.12
   Final Test: 0.00 ± 0.00
```
