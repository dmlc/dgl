# GCN on ogbn-arxiv

Requires DGL 0.5 or later versions.

Run `gcn.py` with `--use-linear` and `use-labels` enabled and you should directly see the result.

```bash
python3 gcn.py --use-linear --use-labels
```

## Usage

```
usage: GCN on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--use-linear]
                         [--lr LR] [--n-layers N_LAYERS] [--n-hidden N_HIDDEN] [--dropout DROPOUT] [--wd WD]
                         [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu.
  --gpu GPU             GPU device ID.
  --n-runs N_RUNS
  --n-epochs N_EPOCHS
  --use-labels          Use labels in the training set as input features.
  --use-linear          Use linear layers.
  --lr LR
  --n-layers N_LAYERS
  --n-hidden N_HIDDEN
  --dropout DROPOUT
  --wd WD
  --log-every LOG_EVERY
  --plot-curves
```

## Results

Here are the results over 10 runs.

|            |       GCN       |   GCN+linear    |   GCN+labels    | GCN+linear+labels |
|------------|:---------------:|:---------------:|:---------------:|:-----------------:|
| Val acc    | 0.7361 ± 0.0009 | 0.7397 ± 0.0010 | 0.7399 ± 0.0008 |  0.7442 ± 0.0012  |
| Test acc   | 0.7246 ± 0.0021 | 0.7270 ± 0.0016 | 0.7259 ± 0.0006 |  0.7306 ± 0.0024  |
| Parameters |     109608      |     218152      |     119848      |      238632       |
