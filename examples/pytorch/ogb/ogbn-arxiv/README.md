# DGL examples for ogbn-arxiv

Requires DGL 0.5 or later versions.

### GCN

Run `gcn.py` with `--use-linear` and `--use-labels` enabled and you should directly see the result.

```bash
python3 gcn.py --use-linear --use-labels
```

### GAT

Run `gat.py` with `--use-labels` enabled and you should directly see the result.

```bash
python3 gat.py --use-norm --use-labels
```

## Usage & Options

### GCN

```
usage: GCN on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--use-linear]
                         [--lr LR] [--n-layers N_LAYERS] [--n-hidden N_HIDDEN] [--dropout DROPOUT] [--wd WD]
                         [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS
  --n-epochs N_EPOCHS
  --use-labels          Use labels in the training set as input features. (default: False)
  --use-linear          Use linear layer. (default: False)
  --lr LR
  --n-layers N_LAYERS
  --n-hidden N_HIDDEN
  --dropout DROPOUT
  --wd WD
  --log-every LOG_EVERY
  --plot-curves
```

### GAT

```
usage: GAT on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--use-norm]
                         [--lr LR] [--n-layers N_LAYERS] [--n-heads N_HEADS] [--n-hidden N_HIDDEN] [--dropout DROPOUT]
                         [--attn_drop ATTN_DROP] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS
  --n-epochs N_EPOCHS
  --use-labels          Use labels in the training set as input features. (default: False)
  --use-norm            Use symmetrically normalized adjacency matrix. (default: False)
  --lr LR
  --n-layers N_LAYERS
  --n-heads N_HEADS
  --n-hidden N_HIDDEN
  --dropout DROPOUT
  --attn_drop ATTN_DROP
  --wd WD
  --log-every LOG_EVERY
  --plot-curves
```

## Results

Here are the results over 10 runs.

|             |       GCN       |   GCN+linear    |   GCN+labels    | GCN+linear+labels |   GAT*+labels   |
|-------------|:---------------:|:---------------:|:---------------:|:-----------------:|:---------------:|
| Val acc     | 0.7361 ± 0.0009 | 0.7397 ± 0.0010 | 0.7399 ± 0.0008 |  0.7442 ± 0.0012  | 0.7504 ± 0.0006 |
| Test acc    | 0.7246 ± 0.0021 | 0.7270 ± 0.0016 | 0.7259 ± 0.0006 |  0.7306 ± 0.0024  | 0.7365 ± 0.0011 |
| #Parameters |     109608      |     218152      |     119848      |      238632       |     1628440     |
