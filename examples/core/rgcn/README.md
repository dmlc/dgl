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

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) |
| ------------ | ------------- | ----------- | ------------- | ------------------------ |
| ~1.1GB       | ~5GB          | 0           |  0GB          | ~243s                    |
| ~1.1GB       | ~3GB          | 1           |  4.4GB        | ~81s                     |

### Accuracies
```
Epoch: 01, Loss: 2.3302, Valid: 47.76%, Test: 46.58%
Epoch: 02, Loss: 1.5486, Valid: 48.31%, Test: 47.12%
Epoch: 03, Loss: 1.1469, Valid: 46.43%, Test: 45.18%
Test accuracy 45.1227
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

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) |
| ------------ | ------------- | ----------- | ------------- | ------------------------ |
| ~404GB       | ~60GB         | 0           |  0GB          | ~216s                    |
| ~404GB       | ~60GB         | 1           |  7GB          | ~157s                    |

### Accuracies
```
Epoch: 01, Loss: 2.0798, Valid: 52.04%
Epoch: 02, Loss: 1.8652, Valid: 54.51%
Epoch: 03, Loss: 1.8175, Valid: 53.71%
```
