# DGL for ogbn-proteins

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

## GAT

### Usage

For the best score, run `gat.py` and you should directly see the result.

```bash
python3 gat.py
```

For the score of `GAT (no-attn-dst)`, run `gat.py` with `--no-attn-dst` enabled and you should directly see the result.

```bash
python3 gat.py --no-attn-dst
```

### Results

Here are the results over 10 runs.

|      Method       | Validation ROC-AUC |  Test ROC-AUC   | #Parameters |
|:-----------------:|:------------------:|:---------------:|:-----------:|
|        GAT        |  0.9194 ± 0.0003   | 0.8682 ± 0.0021 |  2,475,232  |
| GAT (no-attn-dst) |  0.9192 ± 0.0007   | 0.8677 ± 0.0015 |  2,460,352  |

