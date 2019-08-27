Topology Adaptive Graph Convolutional networks (TAGCN)
============

- Paper link: [https://arxiv.org/abs/1710.10370](https://arxiv.org/abs/1710.10370)

Dependencies
------------
- MXNet nightly build
- requests

``bash
pip install mxnet --pre
pip install requests
``

Results
-------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
DGLBACKEND=mxnet python3 train.py --dataset cora --gpu 0 --self-loop
```

* cora: ~0.820 (paper: 0.833)
* citeseer: ~0.702 (paper: 0.714)
* pubmed: ～0.798 (paper: 0.811)