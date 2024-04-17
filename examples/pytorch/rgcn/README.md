# Relational-GCN

* Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
- rdflib
- torchmetrics 0.11.4

Install as follows:
```bash
pip install rdflib
pip install torchmetrics==0.11.4
```

How to run
-------

### Entity Classification

Run with the following for entity classification (available datasets: aifb (default), mutag, bgs, and am)
```bash
python3 entity.py --dataset aifb
```

For mini-batch training, run with the following (available datasets are the same as above)
```bash
python3 entity_sample.py --dataset aifb
```
For multi-gpu training (with sampling), run with the following (same datasets and GPU IDs separated by comma)
```bash
python3 entity_sample_multi_gpu.py --dataset aifb --gpu 0,1
```

### Link Prediction

Run with the following for link prediction on dataset FB15k-237 with filtered-MRR

```bash
python link.py
```
> **_NOTE:_** By default, we use uniform edge sampling instead of neighbor-based edge sampling as in [author's code](https://github.com/MichSchli/RelationPrediction). In practice, we find that it can achieve similar MRR.


Summary
-------

### Entity Classification

| Dataset       | Full-graph | Mini-batch
| ------------- | -------    |  ------
| aifb          | ~0.85      | ~0.82
| mutag         | ~0.70      | ~0.50
| bgs           | ~0.86      | ~0.64
| am            | ~0.78      | ~0.42

### Link Prediction
| Dataset       | Best MRR
| ------------- | -------
| FB15k-237     | ~0.2397
