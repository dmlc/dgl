# Relational-GCN

* Paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
* Tensorflow 2.2+
* requests
* rdflib
* pandas

```
pip install requests tensorflow rdflib pandas
export DGLBACKEND=tensorflow
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification
AIFB: accuracy 92.78% (5 runs, DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 71.47% (5 runs, DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 93.10% (5 runs, DGL n-base=25), 83.10% (paper n-base=40)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 25 --testing --gpu 0
```
