# Relational-GCN

* Paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
* PyTorch 0.4.1+
* requests
* rdflib
* pandas

```
pip install requests torch rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification
AIFB: accuracy 97.22% (DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 75% (DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 82.76% (DGL), 83.10% (paper)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --relabel
```

AM: accuracy 87.37% (DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --n-bases=40 --n-hidden=10 --l2norm=5e-4 --testing
```

#### Entity Classification with minibatch
AIFB: accuracy avg(5 runs) 94.44%, best 94.44% (DGL)
```
python3 entity_classify_hetero_mb.py -d aifb --testing --gpu 0 --fanout=20 --batch-size 128
```

MUTAG: accuracy avg(5 runs) 69.41%, best 73.53% (DGL)
```
python3 entity_classify_hetero_mb.py -d mutag --l2norm 0.00048 --n-bases 10 --testing --gpu 0 --lr=0.005 --batch-size=64 --fanout=32 --n-epochs=30
```

BGS: accuracy avg(5 runs) 92.41%, best 93.10% (DGL)
```
python3 entity_classify_hetero_mb.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout 40 --lr 0.02 --dropout 0.1 --n-epochs=60 --batch-size=128
```

AM: accuracy avg(5 runs) 87.76%, best 89.39% (DGL)
```
python3 entity_classify_hetero_mb.py -d am --l2norm 2e-4 --n-bases 12 --testing --gpu 0 --fanout 35 --batch-size 256 --lr 0.025 --n-hidden 16 --use-self-loop --n-epochs=30
```

#### Entity Classification with multi-gpu minibatch
AM: accuracy avg(5 runs) 87.76%, best 89.39% (DGL)
```
python3 entity_classify_hetero_mb.py -d am --l2norm 2e-4 --n-bases 12 --testing --gpu 0,1,2,3 --fanout 35 --batch-size 256 --lr 0.025 --n-hidden 16 --use-self-loop --n-epochs=30
```

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0 --raw
```
FB15k-237: Filtered-MRR 0.2044
```
python3 link_predict.py -d FB15k-237 --gpu 0 --filtered
```

#### Link Prediction with minibatch
wn18:
Average MRR: 0.6548580399573246
Average MR: 92.5784
Average HITS@1: 0.4874
Average HITS@3: 0.7998
Average HITS@10: 0.952
'''
python3 link_predict_hetero_mb.py --low-mem --lr 0.00165 --n-bases 2 --n-layers 1 --n-epochs 50 --fanout -1 --batch-size 512 --regularization-coef 2e-6 --valid-neg-cnt 1000 --test-neg-cnt -1 --use-self-loop --num-worker 4 --n-hidden 200 --dropout 0.2 --dataset wn18
'''
