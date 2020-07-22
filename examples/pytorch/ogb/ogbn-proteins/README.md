# DGL for ogbn-proteins

## Models
[MWE-GCN and MWE-DGCN](https://cims.nyu.edu/~chenzh/files/GCN_with_edge_weights.pdf) are GCN models designed for graphs whose edges contain multi-dimensional edge weights that indicate the strengths of the relations represented by the edges.

## Dependencies
- DGL 0.4.3
- PyTorch 1.4.0
- OGB 1.2.0
- Tensorboard 2.1.1

## Usage

To use MWE-GCN:
```python
python main_proteins_full_dgl.py --model MWE-GCN
```

To use MWE-DGCN:
```python
python main_proteins_full_dgl.py --model MWE-DGCN
```

Additional optional arguments include 'rand_seed' (the random seed), 'cuda' (the cuda device number, if available), 'postfix' (a string appended to the saved-model file)

