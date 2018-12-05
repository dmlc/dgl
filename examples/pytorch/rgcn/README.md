# Relational-GCN

### Prerequisites
Two extra python packages are needed for this example: 

- rdflib
- pandas

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

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0
```
