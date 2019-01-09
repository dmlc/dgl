Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo (in Tensorflow):
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).
- Popular pytorch implementation:
  [https://github.com/Diego999/pyGAT](https://github.com/Diego999/pyGAT).

How to run
----------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gat_spmv.py --dataset=cora --gpu=0
```

Results
-------

| Dataset | Test Accuracy | Training speed (epoch time) |
| ------- | ------------- | --------------------------- |
| Cora | 84.0% | TBD |
| Citeseer | 70.5% | TBD |
| Pubmed | 77.3% | TBD |

* All the accuracy numbers are obtained after 200 epochs.
