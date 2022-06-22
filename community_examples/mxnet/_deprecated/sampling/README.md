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
DGLBACKEND=mxnet python3 train.py --model gcn_ns --dataset cora --self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 5000
```

citeseer: test accuracy ~69% with `--num-neighbors 2`, ~70% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_ns --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 5000
```

pubmed: test accuracy ~78% with `--num-neighbors 3`, ~77% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_ns --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000 --test-batch-size 5000
```

reddit: test accuracy ~91% with `--num-neighbors 3` and `--batch-size 1000`, ~93% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_ns --dataset reddit-self-loop --num-neighbors 3 --batch-size 1000 --test-batch-size 5000 --n-hidden 64
```


### Control Variate & Skip Connection
cora: test accuracy ~84% with `--num-neighbors 1`, ~84% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_cv --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000
```

citeseer: test accuracy ~69% with `--num-neighbors 1`, ~70% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_cv --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000
```

pubmed: test accuracy ~79% with `--num-neighbors 1`, ~77% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_cv --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000
```

reddit: test accuracy ~93% with `--num-neighbors 1` and `--batch-size 1000`, ~93% by training on the full graph
```
DGLBACKEND=mxnet python3 train.py --model gcn_cv --dataset reddit-self-loop --num-neighbors 1 --batch-size 10000 --test-batch-size 5000 --n-hidden 64
```

### Control Variate & GraphSAGE-mean

Following [Control Variate](https://arxiv.org/abs/1710.10568), we use the mean pooling architecture GraphSAGE-mean, two linear layers and layer normalization per graph convolution layer.

reddit: test accuracy 96.1% with `--num-neighbors 1` and `--batch-size 1000`, ~96.2% in [Control Variate](https://arxiv.org/abs/1710.10568) with `--num-neighbors 2` and `--batch-size 1000`
```
DGLBACKEND=mxnet python3 train.py --model graphsage_cv --batch-size 1000 --test-batch-size 5000 --n-epochs 50 --dataset reddit --num-neighbors 1 --n-hidden 128 --dropout 0.2 --weight-decay 0
```

### Run multi-processing training

When training a GNN model with multiple processes, there are two steps.

Step 1: run a graph store server separately that loads the reddit dataset with four workers.
```
python3 examples/mxnet/sampling/run_store_server.py --dataset reddit --num-workers 4
```

Step 2: run four workers to train GraphSage on the reddit dataset.
```
python3 ../incubator-mxnet/tools/launch.py -n 4 -s 1 --launcher local python3 multi_process_train.py --model graphsage_cv --batch-size 2500 --test-batch-size 5000 --n-epochs 1 --graph-name reddit --num-neighbors 1 --n-hidden 128 --dropout 0.2 --weight-decay 0
```
