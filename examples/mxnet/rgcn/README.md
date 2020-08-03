# Relational-GCN

* Paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
Two extra python packages are needed for this example:

- MXNet nightly build
- requests
- rdflib
- pandas

```bash
pip install mxnet --pre
pip install requests rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification
AIFB: accuracy 97.22% (5 runs, DGL), 95.83% (paper)
```
DGLBACKEND=mxnet python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 70.59% (5 runs, DGL), 73.23% (paper)
```
DGLBACKEND=mxnet python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

BGS: accuracy 86.21% (5 runs, DGL, n-basese=20), 83.10% (paper)
```
DGLBACKEND=mxnet python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 20 --testing --gpu 0
```
