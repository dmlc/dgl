Simple Graph Convolution (SGC)
============

- Paper link: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- Author's code repo: [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC). 

Dependencies
------------
- MXNET 1.5+
- requests

``bash
pip install torch requests
``

Codes
-----
The folder contains an implementation of SGC (`sgc.py`).

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
DGLBACKEND=mxnet python3 sgc.py --dataset cora --gpu 0
DGLBACKEND=mxnet python3 sgc.py --dataset citeseer --weight-decay 5e-5 --n-epochs 150 --bias --gpu 0
DGLBACKEND=mxnet python3 sgc.py --dataset pubmed --weight-decay 5e-5 --bias --gpu 0
```

On NVIDIA V100

* cora: 0.818 (paper: 0.810)
* citeseer: 0.725 (paper: 0.719)
* pubmed: 0.788 (paper: 0.789)
