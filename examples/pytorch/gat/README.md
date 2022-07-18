Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (tensorflow implementation):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

How to run
-------

Run with the following for node classification (available datasets: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora
```

Run with the following for graph classification with PPI dataset
```bash
python3 train_ppi.py
```

Summary
-------
* cora: ~0.821
* citeseer: ~0.710
* pubmed: ~0.780
* ppi: ~0.9744
