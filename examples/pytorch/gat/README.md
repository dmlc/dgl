Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

Dependencies
------------
- torch v1.0: the autograd support for sparse mm is only available in v1.0.
- requests
- sklearn

```bash
pip install torch==1.0.0 requests
```

How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora --gpu=0
```

```bash
python3 train.py --dataset=citeseer --gpu=0 --early-stop
```

```bash
python3 train.py --dataset=pubmed --gpu=0 --num-out-heads=8 --weight-decay=0.001 --early-stop
```

```bash
python3 train_ppi.py --gpu=0
```

Results
-------

| Dataset  | Test Accuracy | Time(s) | Baseline#1 times(s) | Baseline#2 times(s) |
| -------- | ------------- | ------- | ------------------- | ------------------- |
| Cora     | 84.02(0.40)   | 0.0113  | 0.0982 (**8.7x**)   | 0.0424 (**3.8x**)   |
| Citeseer | 70.91(0.79)   | 0.0111  | n/a                 | n/a                 |
| Pubmed   | 78.57(0.75)   | 0.0115  | n/a                 | n/a                 |
| PPI      | 0.9836        | n/a     | n/a                 | n/a                 | 

* All the accuracy numbers are obtained after 300 epochs.
* The time measures how long it takes to train one epoch.
* All time is measured on EC2 p3.2xlarge instance w/ V100 GPU.
* Baseline#1: [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
* Baseline#2: [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).
