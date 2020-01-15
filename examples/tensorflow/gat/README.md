Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

Dependencies
------------
- tensorflow 2.1.0+
- requests

```bash
pip install tensorflow requests
```

How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora --gpu=0 --num-hidden 16 --num-heads 16 --epochs 200 --weight-decay 5e-4 --num-out-heads 4
```

```bash
python3 train.py --dataset=citeseer --gpu=0 --early-stop --num-hidden 16 --num-out-heads 8 --num-heads 8
```

```bash
python3 train.py --dataset=pubmed --gpu=0 --early-stop --num-hidden 16 --num-out-heads 8 --num-heads 16
```


Results
-------

| Dataset  | Test Accuracy |
| -------- | ------------- |
| Cora     | 83.8          |
| Citeseer | 70.8          |
| Pubmed   | 77.9          |

* All the accuracy numbers are obtained after 300 epochs.
* The time measures how long it takes to train one epoch.
* All time is measured on EC2 p3.2xlarge instance w/ V100 GPU.
