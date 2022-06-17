# DGL examples for ogbn-arxiv

DGL implementation of GCN and GAT for [ogbn-arxiv](https://ogb.stanford.edu/docs/nodeprop/). Using some of the techniques from *Bag of Tricks for Node Classification with Graph Neural Networks* ([https://arxiv.org/abs/2103.13355](https://arxiv.org/abs/2103.13355)).

Requires DGL 0.5 or later versions.

### GCN

For the best score, run `gcn.py` with `--use-linear` and `--use-labels` enabled and you should directly see the result.

```bash
python3 gcn.py --use-linear --use-labels
```

### GAT

For the score of `GAT(norm. adj.)+labels`, run the following command and you should directly see the result.

```bash
python3 gat.py --use-norm --use-labels --no-attn-dst --edge-drop=0.1 --input-drop=0.1
```

For the score of `GAT(norm. adj.)+label reuse`, run the following command and you should directly see the result.

```bash
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25
```

For the score of `GAT(norm. adj.)+label reuse+C&S`, run the following command and you should directly see the result.

```bash
python3 gat.py --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.3 --input-drop=0.25 --save-pred
python3 correct_and_smooth.py --use-norm
```

## Usage & Options

### GCN

```
usage: GCN on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--use-linear] [--lr LR] [--n-layers N_LAYERS] [--n-hidden N_HIDDEN]
                         [--dropout DROPOUT] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS       running times (default: 10)
  --n-epochs N_EPOCHS   number of epochs (default: 1000)
  --use-labels          Use labels in the training set as input features. (default: False)
  --use-linear          Use linear layer. (default: False)
  --lr LR               learning rate (default: 0.005)
  --n-layers N_LAYERS   number of layers (default: 3)
  --n-hidden N_HIDDEN   number of hidden units (default: 256)
  --dropout DROPOUT     dropout rate (default: 0.75)
  --wd WD               weight decay (default: 0)
  --log-every LOG_EVERY
                        log every LOG_EVERY epochs (default: 20)
  --plot-curves         plot learning curves (default: False)
```

### GAT

```
usage: GAT on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--n-label-iters N_LABEL_ITERS] [--no-attn-dst]
                         [--use-norm] [--lr LR] [--n-layers N_LAYERS] [--n-heads N_HEADS] [--n-hidden N_HIDDEN] [--dropout DROPOUT] [--input-drop INPUT_DROP]
                         [--attn-drop ATTN_DROP] [--edge-drop EDGE_DROP] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS       running times (default: 10)
  --n-epochs N_EPOCHS   number of epochs (default: 2000)
  --use-labels          Use labels in the training set as input features. (default: False)
  --n-label-iters N_LABEL_ITERS
                        number of label iterations (default: 0)
  --no-attn-dst         Don't use attn_dst. (default: False)
  --use-norm            Use symmetrically normalized adjacency matrix. (default: False)
  --lr LR               learning rate (default: 0.002)
  --n-layers N_LAYERS   number of layers (default: 3)
  --n-heads N_HEADS     number of heads (default: 3)
  --n-hidden N_HIDDEN   number of hidden units (default: 250)
  --dropout DROPOUT     dropout rate (default: 0.75)
  --input-drop INPUT_DROP
                        input drop rate (default: 0.1)
  --attn-drop ATTN_DROP
                        attention dropout rate (default: 0.0)
  --edge-drop EDGE_DROP
                        edge drop rate (default: 0.0)
  --wd WD               weight decay (default: 0)
  --log-every LOG_EVERY
                        log every LOG_EVERY epochs (default: 20)
  --plot-curves         plot learning curves (default: False)
```

## Results

Here are the results over at least 10 runs.

|             Method              | Validation Accuracy |  Test Accuracy  | #Parameters |
|:-------------------------------:|:-------------------:|:---------------:|:-----------:|
|               GCN               |   0.7361 ± 0.0009   | 0.7246 ± 0.0021 |   109,608   |
|           GCN+linear            |   0.7397 ± 0.0010   | 0.7270 ± 0.0016 |   218,152   |
|           GCN+labels            |   0.7399 ± 0.0008   | 0.7259 ± 0.0006 |   119,848   |
|        GCN+linear+labels        |   0.7442 ± 0.0012   | 0.7306 ± 0.0024 |   238,632   |
|     GAT(norm. adj.)+labels      |   0.7508 ± 0.0009   | 0.7366 ± 0.0011 |  1,441,580  |
|   GAT(norm. adj.)+label reuse   |   0.7516 ± 0.0008   | 0.7391 ± 0.0012 |  1,441,580  |
| GAT(norm. adj.)+label reuse+C&S |   0.7519 ± 0.0008   | 0.7395 ± 0.0012 |  1,441,580  |
