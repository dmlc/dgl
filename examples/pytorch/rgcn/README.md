## Usage

### Entity Classification
AIFB:
```
python entity_classify.py -d aifb --testing --gpu 2
```

MUTAG:
```
python entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 2
```

BGS: (--relabel required to fit into TitanX GPU)
```
python entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 2 --relabel
```

### Link Prediction
FB15k-237:
```
python link_predict.py -d FB15k-237 --gpu 2
```
