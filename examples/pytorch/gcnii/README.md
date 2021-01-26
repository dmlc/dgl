Simple and Deep Graph Convolutional Networks
============

- Paper link: [Simple and Deep Graph Convolutional Networks](https://arxiv.org/abs/1810.05997)
- Author's code repo: [https://github.com/chennnM/GCNII](https://github.com/chennnM/GCNII). 

Dependencies
------------
- PyTorch 0.4.1+
- requests

``bash
pip install torch requests
``

Code
-----
The folder contains an implementation of GCNII (`gcnii.py`).

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora --gpu 0
```

```bash
python3 train.py --dataset citeseer --gpu 0 --n-layers 32 --n-hidden 256 --lamda 0.6 --dropout 0.7
```

```bash
python3 train.py --dataset pubmed --gpu 0 --n-layers 16 --n-hidden 256 --lamda 0.4 --dropout 0.5 --wd1 5e-4
```

* cora:		85.46(0.44)		paper: 85.5
* citeseer:	73.32(0.63)		paper: 73.4
* pubmed:	80.08(0.54)		paper: 80.3
