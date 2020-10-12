# Composition-GCN

* Paper: [https://arxiv.org/abs/1911.03082](https://arxiv.org/abs/1911.03082)
* Author's codes: [https://github.com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)

### Dependencies
* PyTorch 1.6
* DGL 0.5.1
* pandas
* rdflib

```
pip install dgl torch pandas
```

### Entity Classification
AIFB: accuracy 94.44%, 95.83% (in original [RGCN paper](https://arxiv.org/pdf/1703.06103.pdf))
```
python dgl/examples/pytorch/compgcn/compgcn_fullgraph.py -d aifb --rev_indicator rev_ --hid_dim 64 --num_layers 3 --comp_fn ccorr --drop_out 0.1 --max_epoch 29 --num_basis 5 --testing
```

BGS: accuracy 89.66% (20 runs,DGL)
```
python compgcn_fullgraph.py -d bgs --rev_indicator rev_ --gpu 0 --lr 0.0001 --comp_fn ccorr --max_epoch 20 --num_layers 4 --drop_out 0.1
```

MUTAG: accuracy 79.41% (11 runs, DGL), 85.3% (paper)
```
python compgcn_fullgraph.py -d mutag --rev_indicator rev_ --gpu 0 --lr 0.001 --comp_fn ccorr --max_epoch 11 --num_layers 4 --drop_out 0.1
```

AM: accuracy 75% (11 runs, DGL), 90.6% (paper)
```
python compgcn_fullgraph.py -d am --rev_indicator rev_ --gpu 0 --lr 0.001 --comp_fn ccorr --max_epoch 11 --num_layers 4 --drop_out 0.1
```
