Composition-based Graph Convolutional Networks (CompGCN)
============

- Paper link: [https://arxiv.org/abs/1911.03082](https://arxiv.org/abs/1911.03082)
- Author's code repo: [https://github.com/malllabiisc/CompGCN](https://github.com/malllabiisc/CompGCN). 

Dependencies
------------
- PyTorch 0.4.1+
- requests

``bash
pip install torch requests
``

Codes
-----
The folder contains three implementations of CompGCN:
- `***.py` module.
- `***.py` . 
Modify `train.py` to switch between different implementations.

Results
-------

Run with following (available dataset: "cora", "citeseer", "pubmed")
```bash
python3 train.py --dataset cora --gpu 0 --self-loop
```

* cora: ~0.810 (0.79-0.83) (paper: 0.815)
* citeseer: 0.707 (paper: 0.703)
* pubmed: 0.792 (paper: 0.790)
