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

All experiments use one-hot encoding as featureless input. Best accuracy reported.


AIFB: accuracy 96.11% (5 runs, DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 72.06% (5 runs, DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 91.73% (5 runs, DGL), 83.10% (paper)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

AM: accuracy 88.28% (5 runs, DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

### Entity Classification w/ minibatch training

Accuracy numbers are reported by 5 runs.

AIFB: accuracy best=97.22% avg=94.44%
```
python3 entity_classify_mb.py -d aifb --testing --gpu 0 --fanout=8
```

MUTAG: accuracy best=76.47% avg=67.37%
```
python3 entity_classify_mb.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --batch-size=50 --fanout=8
```

BGS: accuracy best=96.55% avg=91.04%
```
python3 entity_classify_mb.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

AM: accuracy best=89.39% avg=88.55%
```
python3 entity_classify_mb.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

### Offline Inferencing
Trained Model can be exported by providing '--model\_path <PATH>' parameter to entity\_classify.py. And then test\_classify.py can load the saved model and do the testing offline.

AIFB:
```
python3 entity_classify.py -d aifb --testing --gpu 0 --model_path "aifb.pt"
python3 test_classify.py -d aifb --gpu 0 --model_path "aifb.pt"
```

MUTAG:
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --model_path "mutag.pt"
python3 test_classify.py -d mutag --n-bases 30 --gpu 0 --model_path "mutag.pt"
```

BGS:
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --model_path "bgs.pt"
python3 test_classify.py -d bgs --n-bases 40 --gpu 0 --model_path "bgs.pt"
```

AM:
```
python3 entity_classify.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --model_path "am.pt"
python3 test_classify.py -d am --n-bases 40 --gpu 0 --model_path "am.pt"
```
