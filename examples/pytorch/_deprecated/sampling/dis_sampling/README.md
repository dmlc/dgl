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

### Usage Guide

Assume that the user has already launched two instances (`instance_0` & `instance_1`) on AWS EC2, and also these two instances have the correct authority to access each other by TCP/IP protocol. Now we can treat `instance_0` as `Trainer` and `instance_1` as `Sampler`. Then, the user can start the trainer process and sampler process on these two instances separately. We have already provided a set of scripts to start the trainer and sampler process and users just need to change the `--ip` to their own IP address.

Once we start the trainer process, users will see the following logging output:

```
[04:48:20] .../socket_communicator.cc:68: Bind to 127.0.0.1:2049
[04:48:20] .../socket_communicator.cc:74: Listen on 127.0.0.1:2049, wait sender connect ...
```

After that user can start the sampler process. For the sampler instance_0, users can change the `--num-sampler` option to set the number of the sampler. The `sampler.py` script will start `--num-sampler` processes concurrently to maximalize the system utilization. Users can also launch many samplers in parallel across a set of machines. For example, if we have `10` sampler instance and for each instance, we set the `--num-sampler` to `2`, we need to set the `--num-sampler` of the trainer instance to `20`.

### Neighbor Sampling & Skip Connection

#### cora

Test accuracy ~83% with --num-neighbors 2, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc_train.py --dataset cora --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_ns --dataset cora --self-loop --num-neighbors 2 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### citeseer 

Test accuracy ~69% with --num-neighbors 2, ~70% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc_train.py --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_ns --dataset citeseer --self-loop --num-neighbors 2 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### pubmed 

Test accuracy ~76% with --num-neighbors 3, ~77% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_ns_sc_train.py --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_ns --dataset pubmed --self-loop --num-neighbors 3 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

### Control Variate & Skip Connection

#### cora

Test accuracy ~84% with --num-neighbors 1, ~84% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc_train.py --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_cv --dataset cora --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### citeseer

Test accuracy ~69% with --num-neighbors 1, ~70% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc_train.py --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_cv --dataset citeseer --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

#### pubmed

Test accuracy ~77% with --num-neighbors 1, ~77% by training on the full graph

Trainer side:
```
DGLBACKEND=pytorch python3 gcn_cv_sc_train.py --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --test-batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```

Sampler side:
```
OMP_NUM_THREADS=1 DGLBACKEND=pytorch python3 sampler.py --model gcn_cv --dataset pubmed --self-loop --num-neighbors 1 --batch-size 1000000 --ip 127.0.0.1:50051 --num-sampler 1
```
