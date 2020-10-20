# DGL for ogbn-proteins

## MWE-GCN and MWE-DGCN

### Models
[MWE-GCN and MWE-DGCN](https://cims.nyu.edu/~chenzh/files/GCN_with_edge_weights.pdf) are GCN models designed for graphs whose edges contain multi-dimensional edge weights that indicate the strengths of the relations represented by the edges.

### Dependencies
- DGL 0.4.3
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

### Results

Here are the results over 6 runs.

|      |     Val acc     |    Test acc     | #Parameters |
|:----:|:---------------:|:---------------:|:-----------:|
| GAT* | 0.9163 ± 0.0005 | 0.8665 ± 0.0010 |  2,436,304  |

*GAT\* modified the method in GAT paper by removing the "leaky_relu" in calculating attention, thus reducing the amount of calculation: $\text{softmax}(a^T[Wh_i||Wh_j])=\text{softmax}(a^T Wh_i)$  where $\text{j}$ indicates the destination node of an edge.*
