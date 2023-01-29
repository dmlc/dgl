Relational Graph Attention Networks (RGAT)
==============
This is an adaptation of RGCN where graph convolution is replaced with graph attention.

Dependencies
------------
- torchmetrics

Install as follows:
```bash
pip install torchmetrics
```

How to Run
-------

Run with the following for node classification on ogbn-mag dataset
```bash
python train.py
```


Summary
-------
* ogbn-mag (test acc.): ~0.3647
