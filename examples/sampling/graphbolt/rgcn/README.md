# Node classification on heterogeneous graph with RGCN

## Run on `ogbn-mag` dataset

### Command
```
python3 hetero_rgcn.py
```

### Statistics of train/validation/test
Below results are run on AWS EC2 r6idn.metal, 1024GB RAM, 128 vCPUs(Ice Lake 8375C), 0 GPUs.

| Dataset Size | CPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set)      |
| ------------ | ------------- | ------------------------ | ---------------------------    |
| ~1.1GB       | ~5GB          | ~3min                    | ~1min40s + ~0min9s + ~0min7s    |

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

| Dataset Size | CPU RAM Usage | Time Per Epoch(Training) | Time Per Epoch(Inference: train/val/test set) |
| ------------ | ------------- | ------------------------ | ------------------------- |
| ~443GB       | ~140GB        | ~2min45s                 | ~28min25s + ~4min21s + ~2min54s   |


As labels are hidden for test set, test accuray is always **0.00**. Test submission is saved as `y_pred_mag240m_test-dev.npz` under current directory.
```
Final performance: 
All runs:
Highest Train: 49.29 ± 0.85
Highest Valid: 34.69 ± 0.49
  Final Train: 48.14 ± 1.09
   Final Test: 0.00 ± 0.00
```