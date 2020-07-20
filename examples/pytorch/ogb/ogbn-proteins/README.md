# DGL for OGB-proteins

## Dependencies
- DGL 0.4.3
- PyTorch 1.4.0
- OGB 1.2.0

## Usage

To use MWE-GCN:
```python
python main_proteins_full_dgl.py --model MWE_GCN
```

To use MWE-DGCN:
```python
python main_proteins_full_dgl.py --model MWE_DGCN
```

Additional optional arguments include 'rand_seed' (the random seed), 'cuda' (the cuda device number, if available), 'postfix' (a string appended to the saved-model file)

