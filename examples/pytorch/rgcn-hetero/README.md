# Relational-GCN

* Paper: [https://arxiv.org/abs/1703.06103](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

The preprocessing is slightly different from the author's code. We directly load and preprocess
raw RDF data. For AIFB, BGS and AM,
all literal nodes are pruned from the graph. For AIFB, some training/testing nodes
thus become orphan and are excluded from the training/testing set. The resulting graph
has fewer entities and relations. As a reference (numbers include reverse edges and relations):

| Dataset | #Nodes | #Edges | #Relations | #Labeled |
| --- | --- | --- | --- | --- |
| AIFB | 8,285 | 58,086 | 90 | 176 |
| AIFB-hetero | 7,262 | 48,810 | 78 | 176 |
| MUTAG | 23,644 | 148,454 | 46 | 340 |
| MUTAG-hetero | 27,163 | 148,100 | 46 | 340 |
| BGS | 333,845 | 1,832,398 | 206 | 146 |
| BGS-hetero | 94,806 | 672,884 | 96 | 146 |
| AM | 1,666,764 | 11,976,642 | 266 | 1000 |
| AM-hetero | 881,680 | 5,668,682 | 96 | 1000 |

### Dependencies
* PyTorch 1.0+
* requests
* rdflib

```
pip install requests torch rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification

(all experiments use one-hot encoding as featureless input)

AIFB: accuracy 97.22% (DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 73.53% (DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 93.10% (DGL), 83.10% (paper)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

AM: accuracy 91.41% (DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```
