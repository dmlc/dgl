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
AIFB: accuracy 92.59% (3 runs, DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 72.55% (3 runs, DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 89.66% (3 runs, DGL), 83.10% (paper)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

AM: accuracy 89.73% (3 runs, DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --n-bases=40 --n-hidden=10 --l2norm=5e-4 --testing
```

### Entity Classification with minibatch
AIFB: accuracy avg(5 runs) 90.56%, best 94.44% (DGL)
```
python3 entity_classify_mp.py -d aifb --testing --gpu 0 --fanout='20,20' --batch-size 128
```

MUTAG: accuracy avg(5 runs) 66.77%, best 69.12% (DGL)
```
python3 entity_classify_mp.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --batch-size 256 --use-self-loop --n-epochs 40
```

BGS: accuracy avg(5 runs) 91.72%, best 96.55% (DGL)
```
python3 entity_classify_mp.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '40,40' --n-epochs=40 --batch-size=128
```

AM: accuracy avg(5 runs) 88.28%, best 90.40% (DGL)
```
python3 entity_classify_mp.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '35,35' --batch-size 256 --lr 1e-2 --n-hidden 16 --use-self-loop --n-epochs=40
```

### Entity Classification on OGBN-MAG
Test-bd: P3-8xlarge

OGBN-MAG accuracy 46.22
```
python3 entity_classify_mp.py -d ogbn-mag --testing --fanout='25,30' --batch-size 512 --n-hidden 64 --lr 0.01 --num-worker 0 --eval-batch-size 8 --low-mem --gpu 0,1,2,3,4,5,6,7 --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --mix-cpu-gpu --node-feats --layer-norm
```

OGBN-MAG without node-feats 43.24
```
python3 entity_classify_mp.py -d ogbn-mag --testing --fanout='25,25' --batch-size 256 --n-hidden 64 --lr 0.01 --num-worker 0 --eval-batch-size 8 --low-mem --gpu 0,1,2,3,4,5,6,7 --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --mix-cpu-gpu --layer-norm
```

Test-bd: P2-8xlarge

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol raw
```
FB15k-237: Filtered-MRR 0.2044
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol filtered
```
