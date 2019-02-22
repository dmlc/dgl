# Stochastic Training for Graph Convolutional Networks

* Paper: [Control Variate](https://arxiv.org/abs/1710.10568)
* Paper: [Skip Connection](https://arxiv.org/abs/1809.05343)
* Author's code: [https://github.com/thu-ml/stochastic_gcn](https://github.com/thu-ml/stochastic_gcn)

### Dependencies

- MXNet nightly build

```bash
pip install mxnet --pre
```

### Neighbor Sampling & Skip Connection
cora: test accuracy ~83% with `--num-neighbors 2`, ~84% by training on the full graph
```
DGLBACKEND=mxnet python gcn_ns_sc.py --dataset cora --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

citeseer: test accuracy ~69% with `--num-neighbors 2`, ~70% by training on the full graph
```
DGLBACKEND=mxnet python gcn_ns_sc.py --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

pubmed: test accuracy ~76% with `--num-neighbors 3`, ~77% by training on the full graph
```
DGLBACKEND=mxnet python gcn_ns_sc.py --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

reddit: test accuracy ~91% with `--num-neighbors 2` and `--batch-size 1000`, ~93% by training on the full graph
```
DGLBACKEND=mxnet python gcn_ns_sc.py --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 500 --n-hidden 64
```


### Control Variate & Skip Connection
cora: test accuracy ~84% with `--num-neighbors 1`, ~84% by training on the full graph
```
DGLBACKEND=mxnet python gcn_cv_sc.py --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

citeseer: test accuracy ~69% with `--num-neighbors 1`, ~70% by training on the full graph
```
DGLBACKEND=mxnet python gcn_cv_sc.py --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

pubmed: test accuracy ~77% with `--num-neighbors 1`, ~77% by training on the full graph
```
DGLBACKEND=mxnet python gcn_cv_sc.py --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --gpu 0
```

reddit: test accuracy ~93% with `--num-neighbors 1` and `--batch-size 1000`, ~93% by training on the full graph
```
DGLBACKEND=mxnet python gcn_cv_sc.py --dataset reddit-self-loop --num-neighbors 1 --batch-size 1000 --test-batch-size 500 --n-hidden 64
```

### Control Variate & GraphSAGE-mean

Following [Control Variate](https://arxiv.org/abs/1710.10568), we use the mean pooling architecture GraphSAGE-mean, two linear layers and layer normalization per graph convolution layer.

reddit: test accuracy 96.1% with `--num-neighbors 1` and `--batch-size 1000`, ~96.2% in [Control Variate](https://arxiv.org/abs/1710.10568) with `--num-neighbors 2` and `--batch-size 1000`
```
DGLBACKEND=mxnet python graphsage_cv.py --batch-size 1000 --test-batch-size 500 --n-epochs 50 --dataset reddit --num-neighbors 1 --n-hidden 128 --dropout 0.2 --weight-decay 0
```

