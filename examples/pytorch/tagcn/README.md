Topology Adaptive Graph Convolutional networks (TAGCN)
============

- Paper link: [https://arxiv.org/abs/1710.10370](https://arxiv.org/abs/1710.10370)

Dependencies
------------
- PyTorch 0.4.1+
- requests

``bash
pip install torch requests
``


Codes
-----


Results
-------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --gpu 0 --self-loop
```

* cora: ~0.810 (0.79-0.83) (paper: 0.714)
* citeseer: 0.707 (paper: 0.833)
* pubmed: 0.792 (paper: 0.811)