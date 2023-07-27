Graph Attention Networks v2 (GATv2)
============

- Paper link: [How Attentive are Graph Attention Networks?](https://arxiv.org/pdf/2105.14491.pdf)
- Author's code repo: [https://github.com/tech-srl/how_attentive_are_gats](https://github.com/tech-srl/how_attentive_are_gats).
- Annotated implemetnation: [https://nn.labml.ai/graphs/gatv2/index.html]

Dependencies
------------
- torch
- requests
- scikit-learn

How to run
----------

Run with following:

```bash
python3 train.py --dataset=cora
```

```bash
python3 train.py --dataset=citeseer
```

```bash
python3 train.py --dataset=pubmed
```

Results
-------

| Dataset  | Test Accuracy |
| -------- | ------------- |
| Cora     |  82.10        |
| Citeseer |  70.00        |
| Pubmed   |  77.2         |

* All the accuracy numbers are obtained after 200 epochs.
