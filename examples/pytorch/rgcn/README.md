# Relational-GCN

* Paper: [Modeling Relational Data with Graph Convolutional Networks](https://arxiv.org/abs/1703.06103)
* Author's code for entity classification: [https://github.com/tkipf/relational-gcn](https://github.com/tkipf/relational-gcn)
* Author's code for link prediction: [https://github.com/MichSchli/RelationPrediction](https://github.com/MichSchli/RelationPrediction)

### Dependencies
* PyTorch 1.10
* rdflib
* pandas
* tqdm
* TorchMetrics

```
pip install rdflib pandas
```

Example code was tested with rdflib 4.2.2 and pandas 0.23.4

### Entity Classification

For AIFB, MUTAG, BGS and AM,
```
python entity.py -d aifb --wd 0 --gpu 0
python entity.py -d mutag --n-bases 30 --gpu 0
python entity.py -d bgs --n-bases 40 --gpu 0
python entity.py -d am --n-bases 40 --n-hidden 10 --gpu 0
```

### Entity Classification with minibatch

For AIFB, MUTAG, BGS and AM,
```
python entity_sample.py -d aifb --wd 0 --gpu 0 --fanout='20,20' --batch-size 128
python entity_sample.py -d mutag --n-bases 30 --gpu 0 --batch-size 64 --fanout='-1,-1' --use-self-loop --n-epochs 20 --dropout 0.5
python entity_sample.py -d bgs --n-bases 40 --gpu 0 --fanout='-1,-1'  --n-epochs=16 --batch-size=16 --dropout 0.3
python entity_sample.py -d am --n-bases 40 --gpu 0 --fanout='35,35' --batch-size 64 --n-hidden 16 --use-self-loop --n-epochs=20 --dropout 0.7
```

### Entity Classification on multiple GPUs

To use multiple GPUs, replace `entity_sample.py` with `entity_sample_multi_gpu.py` and specify
multiple GPU IDs separated by comma, e.g., `--gpu 0,1`.

### Link Prediction
FB15k-237 in RAW-MRR
```
python link.py --gpu 0 --eval-protocol raw
```
FB15k-237 in Filtered-MRR
```
python link.py --gpu 0 --eval-protocol filtered
```
