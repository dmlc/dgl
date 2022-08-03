# Relational-GCN

* Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
- rdflib
- torchmetrics

Install as follows:
```bash
pip install rdflib
pip install torchmetrics
```

How to run
-------

### Entity Classification

Run with the following for entity classification (available datasets: aifb (default), mutag, bgs, and am)
```bash
python3 entity.py --dataset aifb
```

For minibatch training, run with the following (available datasets are the same as above)
```bash
python3 entity_sample.py --dataset aifb
```
For multi-gpu training (with sampling), run with the following (same datasets and GPU IDs separated by comma) 
```bash
python3 entity_sample_multi_gpu.py --dataset aifb --gpu 0,1
```


### Link Prediction
FB15k-237 in RAW-MRR
```
python link.py --gpu 0 --eval-protocol raw
```
FB15k-237 in Filtered-MRR
```
python link.py --gpu 0 --eval-protocol filtered
```

Summary 
-------
### Entity Classification

| Dataset       | Result 
| ------------- | -------
| aifb        | ~84.3    
| mutag       | ~69.7   
| bgs         | ~91.4    
| am          | ~78.5
