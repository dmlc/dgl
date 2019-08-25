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

Results
-------
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora --gpu 0 --self-loop
```

* cora: ~0.812 (0.804-0.823) (paper: 0.833)
* citeseer: ~0.715 (paper: 0.714)
* pubmed: ～0.794 (paper: 0.811)