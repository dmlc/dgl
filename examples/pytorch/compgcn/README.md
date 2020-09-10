# Composition-GCN

* Paper: [https://arxiv.org/abs/1911.03082](https://arxiv.org/abs/1911.03082)
* Author's codes: [https://github.com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN)

### Dependencies
* PyTorch 1.6
* DGL 0.5.1
* pandas

```
pip install dgl torch pandas
```

### Entity Classification

AM: accuracy 89.73% (3 runs, DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --n-bases=40 --n-hidden=10 --l2norm=5e-4 --testing
```

### Entity Classification with minibatch

AM: accuracy avg(5 runs) 88.28%, best 90.40% (DGL)
```
python3 entity_classify_mp.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '35,35' --batch-size 256 --lr 1e-2 --n-hidden 16 --use-self-loop --n-epochs=40
```

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol raw
```
FB15k-237: Filtered-MRR 0.2044
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol filtered
```
