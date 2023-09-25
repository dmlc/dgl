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
Below results are roughly collected from an AWS EC2 **g4dn.metal**, 384GB RAM, 96 vCPUs(Cascade Lake P-8259L), 8 NVIDIA T4 GPUs.

| Dataset Size | CPU RAM Usage | Num of GPUs | GPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ----------- | ---------- | --------- | ---------------------------    |
| ~1.1GB       | ~5GB          | 0           |  0GB       | ~4min5s   | ~2min7s + ~0min12s + ~0min8s   |
| ~1.1GB       | ~4.3GB        | 1           |  4.7GB     | ~1min18s  | ~1min54s + ~0min12s + ~0min8s  |

### Accuracies
```
Final performance: 
All runs:
Highest Train: 49.29 ± 0.85
Highest Valid: 34.69 ± 0.49
  Final Train: 48.14 ± 1.09
   Final Test: 33.65 ± 0.63
```

## Run on `ogb-lsc-mag240m` dataset

### Command
```
python3 hetero_rgcn.py --dataset ogb-lsc-mag240m --runs 2
```

### Statistics of train/validation/test
Below results are run on AWS EC2 r6idn.metal, 1024GB RAM, 128 vCPUs(Ice Lake 8375C), 0 GPUs.

| Dataset Size | Peak CPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set) |
| ------------ | ------------- | ------------------------ | ------------------------- |
| ~404GB       | ~110GB        | ~2min45s                 | ~28min25s + ~4min21s + ~2min54s   |


As labels are hidden for test set, test accuray is always **0.00**. Test submission is saved as `y_pred_mag240m_test-dev.npz` under current directory.

As we can see from above table, the time per epoch is quite close to the one in `ogbn-mag`. This is due to no embedding layer is applied for `ogb-lsc-mag240m`. All required node features are generated in advance.
```
Final performance: 
All runs:
Highest Train: 54.75 ± 0.29
Highest Valid: 52.08 ± 0.09
  Final Train: 54.75 ± 0.29
   Final Test: 0.00 ± 0.00
```