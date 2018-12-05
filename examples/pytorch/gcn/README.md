Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
implemented with Tensorflow for the paper. 

Codes
-----
The folder contains two implementations of GCN. `gcn_batch.py` uses user-defined
message and reduce functions. `gcn_spmv.py` uses DGL's builtin functions so
SPMV optimization could be applied.

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python gcn_spmv.py --dataset cora --gpu 0
```

* cora: ~0.810 (0.79-0.83) (paper: 0.815)
* citeseer: 0.707 (paper: 0.703)
* pubmed: 0.792 (paper: 0.790)
