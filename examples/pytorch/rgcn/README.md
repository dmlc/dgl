# Relational-GCN

### Prerequisites
Two extra python packages are needed for this example: rdflib, pandas

### Entity Classification
AIFB:
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG:
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS:
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --relabel
```

### Link Prediction
FB15k-237:
```
python3 link_predict.py -d FB15k-237 --gpu 0
```
