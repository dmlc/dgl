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
DGLBACKEND=tensorflow
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


Results
-------

| Dataset  | Test Accuracy | Baseline (paper) |
| -------- | ------------- | ---------------- |
| Cora     | 84.2          | 83.0(+-0.7)      |
| Citeseer | 70.9          | 72.5(+-0.7)      |
| Pubmed   | 78.5          | 79.0(+-0.3)      |

* All the accuracy numbers are obtained after 200 epochs.
* All time is measured on EC2 p3.2xlarge instance w/ V100 GPU.
