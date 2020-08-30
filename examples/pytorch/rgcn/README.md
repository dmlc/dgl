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
AIFB: accuracy 92.59% (3 runs, DGL), 95.83% (paper)
```
python3 entity_classify.py -d aifb --testing --gpu 0
```

MUTAG: accuracy 72.55% (3 runs, DGL), 73.23% (paper)
```
python3 entity_classify.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0
```

BGS: accuracy 89.66% (3 runs, DGL), 83.10% (paper)
```
python3 entity_classify.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0
```

AM: accuracy 89.73% (3 runs, DGL), 89.29% (paper)
```
python3 entity_classify.py -d am --n-bases=40 --n-hidden=10 --l2norm=5e-4 --testing
```

### Entity Classification with minibatch
AIFB: accuracy avg(5 runs) 90.56%, best 94.44% (DGL)
```
python3 entity_classify_mp.py -d aifb --testing --gpu 0 --fanout='20,20' --batch-size 128
```

MUTAG: accuracy avg(5 runs) 66.77%, best 69.12% (DGL)
```
python3 entity_classify_mp.py -d mutag --l2norm 5e-4 --n-bases 30 --testing --gpu 0 --batch-size 256 --use-self-loop --n-epochs 40
```

BGS: accuracy avg(5 runs) 91.72%, best 96.55% (DGL)
```
python3 entity_classify_mp.py -d bgs --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '40,40' --n-epochs=40 --batch-size=128
```

AM: accuracy avg(5 runs) 88.28%, best 90.40% (DGL)
```
python3 entity_classify_mp.py -d am --l2norm 5e-4 --n-bases 40 --testing --gpu 0 --fanout '35,35' --batch-size 256 --lr 1e-2 --n-hidden 16 --use-self-loop --n-epochs=40
```

### Entity Classification on OGBN-MAG
Test-bd: P3-8xlarge

OGBN-MAG accuracy 46.22
```
python3 entity_classify_mp.py -d ogbn-mag --testing --fanout='25,30' --batch-size 512 --n-hidden 64 --lr 0.01 --num-worker 0 --eval-batch-size 8 --low-mem --gpu 0,1,2,3,4,5,6,7 --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --mix-cpu-gpu --node-feats --layer-norm
```

OGBN-MAG without node-feats 43.24
```
python3 entity_classify_mp.py -d ogbn-mag --testing --fanout='25,25' --batch-size 256 --n-hidden 64 --lr 0.01 --num-worker 0 --eval-batch-size 8 --low-mem --gpu 0,1,2,3,4,5,6,7 --dropout 0.5 --use-self-loop --n-bases 2 --n-epochs 3 --mix-cpu-gpu --layer-norm
```

Test-bd: P2-8xlarge

### Link Prediction
FB15k-237: MRR 0.151 (DGL), 0.158 (paper)
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol raw
```
FB15k-237: Filtered-MRR 0.2044
```
python3 link_predict.py -d FB15k-237 --gpu 0 --eval-protocol filtered
```

### Link Prediction with Minibatch and Neighbor Sampling
#### WM18
```
Average MRR: 0.820
Average MR: 7.2853
Average HITS@1: 0.7105
Average HITS@3: 0.923
Average HITS@10: 0.9758
```
```
python3 link_predict_mp.py --low-mem --lr 0.00165 --n-bases 2 --n-layers 1 --n-epochs 20 --fanout -1 --batch-size 256 --regularization-coef 1e-4 --valid-neg-cnt 1000 --test-neg-cnt -1 --use-self-loop --num-worker 4 --n-hidden 200 --dropout 0.2 --dataset wn18
```

#### FB15k-237
filtered MRR 0.264 (DGL), 0.249 (paper)
```
Average MRR: 0.26377048539984593
Average MR: 161.4138326981335
Average HITS@1: 0.1804211863578618
Average HITS@3: 0.28664614482556433
Average HITS@10: 0.4319603244405355
```
```
python3 link_predict_mp.py --lr 0.002 --n-bases 100 --n-layers 2 --n-epochs 2000 --batch-size 30000 --regularization-coef 0.01 --valid-neg-cnt 1000 --test-neg-cnt -1 --use-self-loop --num-worker 4 --n-hidden 500 --dropout 0.4 --dataset FB15k-237 --sampler=path --chunk-size 20 --global-norm --relation-regularizer bdd --gamma 200.0
```

Raw MRR 0.149 (DGL), 0.156 (paper)
```
Average MRR: 0.14897002375811436
Average MR: 321.6044903742793
Average HITS@1: 0.06591419915958174
Average HITS@3: 0.16400371347600898
Average HITS@10: 0.3168425681618294
```
```
python3 link_predict_mp.py --lr 0.002 --n-bases 100 --n-layers 2 --n-epochs 2000 --batch-size 30000 --regularization-coef 0.01 --valid-neg-cnt 1000 --test-neg-cnt -1 --use-self-loop --num-worker 4 --n-hidden 500 --dropout 0.4 --dataset FB15k-237 --sampler=path --chunk-size 20 --global-norm --relation-regularizer bdd --gamma 200.0 --no-test-filter
```
#### FB15k
filtered MRR 0.473s, 0.696 (paper)

```
Average MRR: 0.4727096785053779
Average MR: 55.634710771783105
Average HITS@1: 0.34635438709349764
Average HITS@3: 0.543515430583535
Average HITS@10: 0.7106109596925734
```
```
python3 link_predict_mp.py --lr 0.005 --n-bases 100 --n-layers 2 --n-epochs 2000 --batch-size 60000 --regularization-coef 0.01 --valid-neg-cnt 1000 --test-neg-cnt -1 --use-self-loop --num-worker 4 --n-hidden 500 --dropout 0.4 --dataset FB15k --sampler=path --chunk-size 10 --global-norm --relation-regularizer bdd --gamma 20.0 --num-test-worker 8 --layer-norm
```
