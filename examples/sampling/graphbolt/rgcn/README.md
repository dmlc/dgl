# Node classification on heterogeneous graph with RGCN

This example aims to demonstrate how to run node classification task on heterogeneous graph with **GraphBolt**. Models are not tuned to achieve the best accuracy yet.

## Run on `ogbn-mag` dataset

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
| ~1.1GB       | ~4.5GB        | 0           |  0GB       | ~3min55s(615it, 2.62it/s)   | ~0min18s(154it, 8.37it/s) + ~0min1s(16it, 8.37it/s) + ~0min1s(11it, 8.88it/s)   |
| ~1.1GB       | ~2GB          | 1           |  4.4GB     | ~58s(615it, 10.43it/s)   | ~0min18s(154it, 8.29it/s) + ~0min1s(16it, 8.46it/s) + ~0min1s(11it, 8.82it/s)   |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 64.66 ± 0.74
Highest Valid: 41.31 ± 0.12
  Final Train: 64.66 ± 0.74
   Final Test: 40.07 ± 0.02
```

## Run on `ogb-lsc-mag240m` dataset

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
| ~404GB       | ~55GB       | 0           |  0GB       | ~3min25s(1087it, 5.29it/s)  | ~2min26s(272it, 1.86it/s) + ~0min20s(34it, 1.62it/s) + ~0min13s(22it, 1.68it/s)   |
| ~404GB       | ~55GB       | 1           |  7GB       | ~2min41s(1087it, 6.73it/s)  | ~1min52s(272it, 2.41it/s) + ~0min17s(34it, 1.93it/s) + ~0min11s(22it, 1.99it/s)  |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 54.43 ± 0.39
Highest Valid: 51.78 ± 0.68
  Final Train: 54.35 ± 0.51
   Final Test: 0.00 ± 0.00
```
