# Composition-GCN

* Paper: [https://arxiv.org/abs/1911.03082](https://arxiv.org/abs/1911.03082)
* Author's codes: [https://github.com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)

### Dependencies
* PyTorch 1.6
* DGL 0.5.1
* pandas

```
pip install dgl torch pandas
```

### Entity Classification
AIFB: accuracy 91.67% (20 runs,DGL), 95.83% (paper)
```
python compgcn_fullgraph.py -d aifb --rev_indicator rev_ --gpu 0 --lr 0.001 --comp_fn ccorr --max_epoch 100 --num_layers 4 --drop_out 0.1
```

BGS: accuracy 89.66% (20 runs,DGL)
```
python compgcn_fullgraph.py -d bgs --rev_indicator rev_ --gpu 0 --lr 0.0001 --comp_fn ccorr --max_epoch 20 --num_layers 4 --drop_out 0.1
```

MUTAG: accuracy 79.41% (11 runs, DGL), 85.3% (paper)
```
python compgcn_fullgraph.py -d mutag --rev_indicator rev_ --gpu 0 --lr 0.001 --comp_fn ccorr --max_epoch 11 --num_layers 4 --drop_out 0.1
```

AM: accuracy **% (11 runs, DGL), 90.6% (paper)
```
python compgcn_fullgraph.py -d am --rev_indicator rev_ --gpu 0 --lr 0.001 --comp_fn ccorr --max_epoch 11 --num_layers 4 --drop_out 0.1
```

### Entity Classification with minibatch

AM: accuracy avg(5 runs) 88.28%, best 90.40% (DGL)
```
python3 entity_classify_mp.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '35,35' --batch-size 256 --lr 1e-2 --n-hidden 16 --use-self-loop --n-epochs=40
```
