Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (tensorflow implementation):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

How to run
-------

Run with the following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```

Summary
-------
* cora: ~0.821
* citeseer: ~0.710
* pubmed: ~0.780
