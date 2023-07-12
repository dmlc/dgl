Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (tensorflow implementation):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

How to run
-------

Run with the following for multiclass node classification (available datasets: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```

> **_NOTE:_**  Users may occasionally run into low accuracy issue (e.g., test accuracy < 0.8) due to overfitting. This can be resolved by adding Early Stopping or reducing maximum number of training epochs.

Summary
-------
* cora: ~0.821
* citeseer: ~0.710
* pubmed: ~0.780
