Relational Graph Attention Networks (RGAT)
==============
This is an adaptation of RGCN where graph convolution is replaced with graph attention.

Dependencies
------------
- torchmetrics 0.11.4

Install as follows:
```bash
pip install torchmetrics==0.11.4
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
