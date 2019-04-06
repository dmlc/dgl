Predict then Propagate: Graph Neural Networks meet Personalized PageRank (APPNP)
============

- Paper link: [Predict then Propagate: Graph Neural Networks meet Personalized PageRank](https://arxiv.org/abs/1810.05997)
- Author's code repo: [https://github.com/klicperajo/ppnp](https://github.com/klicperajo/ppnp). 

Dependencies
------------
- PyTorch 0.4.1+
- requests

``bash
pip install torch requests
``

Codes
-----
The folder contains an implementation of APPNP (`appnp.py`).

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python train.py --dataset cora --gpu 0
```

* cora: 0.8370 (paper: 0.850)
* citeseer: 0.715 (paper: 0.757)
* pubmed: 0.793 (paper: 0.797)
