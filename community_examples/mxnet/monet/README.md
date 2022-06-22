MoNet
=====

- paper link: [Geometric deep learning on graphs and manifolds using mixture model CNNs](https://arxiv.org/pdf/1611.08402.pdf)

Dependencies
============

- MXNet 1.5+

Results
=======

## Citation networks
Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 citation.py --dataset cora --gpu 0
```

- Cora: ~0.814
- Pubmed: ~0.748
