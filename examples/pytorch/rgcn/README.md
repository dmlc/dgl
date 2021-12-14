# Relational-GCN

* Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
* PyTorch 1.10
* rdflib
* pandas
* tqdm

```
pip install rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification
AIFB: accuracy 96.29% (3 runs, DGL), 95.83% (paper)
```
python entity.py -d aifb --l2norm 0 --gpu 0
```

MUTAG: accuracy 72.06% (3 runs, DGL), 73.23% (paper)
```
python entity.py -d mutag --n-bases 30 --gpu 0
```

BGS: accuracy 93.10% (3 runs, DGL), 83.10% (paper)
```
python entity.py -d bgs --n-bases 40 --gpu 0
```

AM: accuracy 89.39% (3 runs, DGL), 89.29% (paper)
```
python entity.py -d am --n-bases 40 --n-hidden 10
```

### Entity Classification with minibatch

AIFB: accuracy avg(5 runs) 90.00%, best 94.44% (DGL)
```
python entity_sample.py -d aifb --l2norm 0 --gpu 0 --fanout='20,20' --batch-size 128
```

MUTAG: accuracy avg(10 runs) 62.94%, best 72.06% (DGL)
```
python entity_sample.py -d mutag --n-bases 30 --gpu 0 --batch-size 64 --fanout "-1, -1" --use-self-loop --dgl-sparse --n-epochs 20 --sparse-lr 0.01 --dropout 0.5
```

BGS: accuracy avg(5 runs) 78.62%, best 86.21% (DGL)
```
python entity_sample.py -d bgs --n-bases 40 --gpu 0 --fanout "-1, -1"  --n-epochs=16 --batch-size=16 --dgl-sparse  --lr 0.01 --sparse-lr 0.05 --dropout 0.3
```

AM: accuracy avg(5 runs) 87.37%, best 89.9% (DGL)
```
python entity_sample.py -d am --n-bases 40 --gpu 0 --fanout '35,35' --batch-size 64 --n-hidden 16 --use-self-loop --n-epochs=20 --dgl-sparse --lr 0.01  --sparse-lr 0.02 --dropout 0.7
```

To use multiple GPUs, replace `entity_sample.py` with `entity_sample_multi_gpu.py` and specify
multiple GPU IDs separated by comma, e.g., `--gpu 0,1`.

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python link.py -d FB15k-237 --gpu 0 --eval-protocol raw
```
FB15k-237: Filtered-MRR 0.2044
```
python link.py -d FB15k-237 --gpu 0 --eval-protocol filtered
```
