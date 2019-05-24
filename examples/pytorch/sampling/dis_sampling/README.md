# Stochastic Training for Graph Convolutional Networks Using Distributed Sampler

* Paper: [Control Variate](https://arxiv.org/abs/1710.10568)
* Paper: [Skip Connection](https://arxiv.org/abs/1809.05343)
* Author's code: [https://github.com/thu-ml/stochastic_gcn](https://github.com/thu-ml/stochastic_gcn)

Dependencies
------------
- PyTorch 0.4.1+
- requests

``bash
pip install torch requests
``

### Neighbor Sampling & Skip Connection

#### cora

Test accuracy ~83% with --num-neighbors 2, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset cora --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset cora --self-loop --num-neighbors 2 --batch-size 1000000 --ip 127.0.0.1:50051
```

#### citeseer 

Test accuracy ~69% with --num-neighbors 2, ~70% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000000 --ip 127.0.0.1:50051
```

#### pubmed 

Test accuracy ~76% with --num-neighbors 3, ~77% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc.py --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000000 --ip 127.0.0.1:50051
```

### Control Variate & Skip Connection

#### cora

Test accuracy ~84% with --num-neighbors 1, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051
```

#### citeseer

Test accuracy ~69% with --num-neighbors 1, ~70% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051
```

#### pubmed

Test accuracy ~77% with --num-neighbors 1, ~77% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc.py --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051
```

