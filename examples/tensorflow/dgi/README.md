Deep Graph Infomax (DGI)
========================

- Paper link: [https://arxiv.org/abs/1809.10341](https://arxiv.org/abs/1809.10341)
- Author's code repo (in Pytorch):
  [https://github.com/PetarV-/DGI](https://github.com/PetarV-/DGI)

Dependencies
------------
- tensorflow 2.1+
- requests

```bash
pip install tensorflow requests
```

How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora --gpu=0 --self-loop
```

```bash
python3 train.py --dataset=citeseer --gpu=0
```

```bash
python3 train.py --dataset=pubmed --gpu=0
```

Results
-------
* cora: ~81.6 (80.9-82.9) (paper: 82.3)
* citeseer: ~70.2 (paper: 71.8)
* pubmed: ~77.2 (paper: 76.8)
