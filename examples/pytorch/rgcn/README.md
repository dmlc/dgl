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

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0
```
