Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo:
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).

Note that the original code is implemented with Tensorflow for the paper.

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gat.py --dataset cora --gpu 0 --num-heads 8
```
