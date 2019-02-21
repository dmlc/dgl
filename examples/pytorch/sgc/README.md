Simple Graph Convolution (SGC)
============

- Paper link: [Simplifying Graph Convolutional Networks](https://arxiv.org/abs/1902.07153)
- Author's code repo: [https://github.com/Tiiiger/SGC](https://github.com/Tiiiger/SGC). 

Dependencies
------------
- PyTorch 0.4.1+
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
python sgc.py --dataset cora --gpu 0
python sgc.py --dataset citeseer --weight-decay 5e-5 --n-epochs 150 --bias --gpu 0
python sgc.py --dataset pubmed --weight-decay 5e-5 --bias --gpu 0
```

On NVIDIA V100

* cora: 0.819 (paper: 0.810), 0.0008s/epoch
* citeseer: 0.725 (paper: 0.719), 0.0008s/epoch
* pubmed: 0.788 (paper: 0.789), 0.0007s/epoch
