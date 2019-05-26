
# Stochastic Training for Graph Convolutional Networks Using Distributed Sampler

* Paper: [Control Variate](https://arxiv.org/abs/1710.10568)
* Paper: [Skip Connection](https://arxiv.org/abs/1809.05343)
* Author's code: [https://github.com/thu-ml/stochastic_gcn](https://github.com/thu-ml/stochastic_gcn)

### Dependencies

- MXNet nightly build

```bash
pip install mxnet --pre
```

### Usage

To run the following demos, you need to start trainer and sampler process on different machines by changing the `--ip`. 

### Neighbor Sampling & Skip Connection

#### cora

Test accuracy ~83% with `--num-neighbors 2`, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_ns --dataset cora --self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 5000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_ns --dataset cora --self-loop --num-neighbors 2 --batch-size 1000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### citeseer 

Test accuracy ~69% with `--num-neighbors 2`, ~70% by training on the full graph

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_ns --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 5000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_ns --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### pubmed

Test accuracy ~78% with `--num-neighbors 3`, ~77% by training on the full graph

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_ns --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000 --test-batch-size 5000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_ns --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### reddit

Test accuracy ~91% with `--num-neighbors 2` and `--batch-size 1000`, ~93% by training on the full graph

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_ns --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --test-batch-size 5000 --n-hidden 64 --ip 127.0.0.1:2049 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_ns --dataset reddit-self-loop --num-neighbors 2 --batch-size 1000 --ip 127.0.0.1:2049 --num-sampler 1
```

### Control Variate & Skip Connection

#### cora

Test accuracy ~84% with `--num-neighbors 1`, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_cv --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_cv --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### citeseer

Test accuracy ~69% with `--num-neighbors 1`, ~70% by training on the full graph

Trainer Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_cv --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_cv --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### pubmed

Trainer Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_cv --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_cv --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### reddit

Test accuracy ~93% with `--num-neighbors 1` and `--batch-size 1000`, ~93% by training on the full graph

Trainer Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model gcn_cv --dataset reddit-self-loop --num-neighbors 1 --batch-size 10000 --test-batch-size 5000 --n-hidden 64 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler Side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model gcn_cv --dataset reddit-self-loop --num-neighbors 1 --batch-size 10000 --ip 127.0.0.1:50051 --num-sampler 1
```

### Control Variate & GraphSAGE-mean

Following [Control Variate](https://arxiv.org/abs/1710.10568), we use the mean pooling architecture GraphSAGE-mean, two linear layers and layer normalization per graph convolution layer.

#### reddit

Test accuracy 96.1% with `--num-neighbors 1` and `--batch-size 1000`, ~96.2% in [Control Variate](https://arxiv.org/abs/1710.10568) with `--num-neighbors 2` and `--batch-size 1000`

Trainer side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/train.py --model graphsage_cv --batch-size 1000 --test-batch-size 5000 --n-epochs 50 --dataset reddit --num-neighbors 1 --n-hidden 128 --dropout 0.2 --weight-decay 0 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=mxnet python3 examples/mxnet/sampling/dis_sampling/sampler.py --model graphsage_cv --batch-size 1000 --dataset reddit --num-neighbors 1 --ip 127.0.0.1:50051 --num-sampler 1
```
