# Node classification on heterogeneous graph with RGCN

This example aims to demonstrate how to run node classification task on heterogeneous graph with **GraphBolt**. Models are not tuned to achieve the best accuracy yet.

## Run on `ogbn-mag` dataset

### Sample on CPU and train/infer on CPU
```
python3 hetero_rgcn.py --dataset ogbn-mag
```

### Sample on CPU and train/infer on GPU
```
python3 hetero_rgcn.py --dataset ogbn-mag --num_gups 1
```

### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.metal**, 384GB RAM, 96 vCPUs(Cascade Lake P-8259L), 8 NVIDIA T4 GPUs(16GB RAM). CPU RAM usage is the peak value of `used` field of `free` command which is a bit rough. Please refer to `RSS`/`USS`/`PSS` which are more accurate. GPU RAM usage is the peak value recorded by `nvidia-smi` command.

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ----------- | ---------- | --------- | ---------------------------    |
| ~1.1GB       | ~5GB          | 0           |  0GB       | ~4min5s   | ~2min7s + ~0min12s + ~0min8s   |
| ~1.1GB       | ~4.3GB        | 1           |  4.7GB     | ~1min18s  | ~1min54s + ~0min12s + ~0min8s  |

| ~1.1GB       | ~4.5GB        | 0           |  0GB       | ~4min14s(615it, 2.41it/s)   | ~0min28s(154it, 5.46it/s) + ~0min2s(16it, 5.48it/s) + ~0min2s(11it, 5.44it/s)   |
| ~1.1GB       | ~2GB          | 0           |  4.4GB     | ~1min15s(615it, 8.11it/s)   | ~0min27s(154it, 5.63it/s) + ~0min2s(16it, 5.90it/s) + ~0min1s(11it, 5.82it/s)   |

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
python3 hetero_rgcn.py --dataset ogb-lsc-mag240m --num_gups 1
```

### Resource usage and time cost
Below results are roughly collected from an AWS EC2 **g4dn.metal**, 384GB RAM, 96 vCPUs(Cascade Lake P-8259L), 8 NVIDIA T4 GPUs(16GB RAM). CPU RAM usage is the peak value of `used` and `buff/cache` field of `free` command which are a bit rough. Please refer to `RSS`/`USS`/`PSS` which are more accurate. GPU RAM usage is the peak value recorded by `nvidia-smi` command.

Infer with full neighbors on GPU is out of memory on `T4(16GB RAM)``. GPUs with larger memory is required such as `A100(40GB RAM)`.
```
Tried to allocate 21.72 GiB (GPU 0; 14.75 GiB total capacity; 12.30 GiB already allocated; 2.02 GiB free; 12.60 GiB reserved in total by PyTorch)
```

| Dataset Size | CPU RAM Usage(used + buff/cache) | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ----------- | ---------- | --------- | ---------------------------    |
| ~404GB       | ~110GB + ~250GB       | 0           |  0GB       | ~5min22s(1087it, 3.37it/s)   | ~35min29s(272it, 7.83s/it) + ~6min9s(34it, 10.87s/it) + ~3min32s(22it, 9.66s/it)   |
| ~404GB       | ~55GB + ~250GB        | 1           |  2.7GB     | ~2min45s(1087it, 6.56it/s)  | ~OOM + ~OOM + ~OOM  |

| ~404GB       | ~55GB + ~317GB       | 0           |  0GB       | ~3min51s(1087it, 4.70it/s)   | ~2min21s(272it, 1.93it/s) + ~0min22s(34it, 1.48it/s) + ~0min14s(22it, 1.51it/s)   |
| ~404GB       | ~55GB + ~317GB       | 1           |  8.7GB     | ~5min10s(1087it, 3.5it/s)  | ~3min34s(272it, 1.27it/s) + ~0min32s(34it, 1.06it/s) + ~0min20s(22it, 1.08it/s)  |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 54.75 ± 0.29
Highest Valid: 52.08 ± 0.09
  Final Train: 54.75 ± 0.29
   Final Test: 0.00 ± 0.00
```