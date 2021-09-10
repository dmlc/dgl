# DGL for ogbn-proteins

## GAT

DGL implementation of GAT for [ogbn-proteins](https://ogb.stanford.edu/docs/nodeprop/). Using some of the techniques from *Bag of Tricks for Node Classification with Graph Neural Networks* ([https://arxiv.org/abs/2103.13355](https://arxiv.org/abs/2103.13355)).

Requires DGL 0.5 or later versions.

### Usage

For the best score, run `gat.py` and you should directly see the result.

```bash
python3 gat.py
```

For the score of `GAT+labels`, run `gat.py` with `--use-labels` enabled and you should directly see the result.

```bash
python3 gat.py --use-labels
```

### Results

Here are the results over 10 runs.

|   Method   | Validation ROC-AUC |  Test ROC-AUC   | #Parameters |
|:----------:|:------------------:|:---------------:|:-----------:|
|    GAT     |  0.9276 ± 0.0007   | 0.8747 ± 0.0016 |  2,475,232  |
| GAT+labels |  0.9280 ± 0.0008   | 0.8765 ± 0.0008 |  2,484,192  |

## MWE-GCN and MWE-DGCN

### Models
[MWE-GCN and MWE-DGCN](https://cims.nyu.edu/~chenzh/files/GCN_with_edge_weights.pdf) are GCN models designed for graphs whose edges contain multi-dimensional edge weights that indicate the strengths of the relations represented by the edges.

### Dependencies
- DGL 0.5.2
- PyTorch 1.4.0
- OGB 1.2.0
- Tensorboard 2.1.1

### Usage

To use MWE-GCN:
```python
python main_proteins_full_dgl.py --model MWE-GCN
```

To use MWE-DGCN:
```python
python main_proteins_full_dgl.py --model MWE-DGCN
```

Additional optional arguments include 'rand_seed' (the random seed), 'cuda' (the cuda device number, if available), 'postfix' (a string appended to the saved-model file)

