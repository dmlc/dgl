Graph Isomorphism Network (GIN)
============

- Paper link: [arXiv](https://arxiv.org/abs/1810.00826) [OpenReview](https://openreview.net/forum?id=ryGs6iA5Km) 
- Author's code repo: [https://github.com/weihua916/powerful-gnns](https://github.com/weihua916/powerful-gnns).

Dependencies
------------
- PyTorch 1.0.1+
- sklearn
- tqdm

``bash
pip install torch sklearn tqdm
``

How to run
----------

An experiment on the GIN in default settings can be run with

```bash
python3 main.py
```

An experiment on the GIN in customized settings can be run with
```bash
python3 main.py [--device 0 | --disable-cuda] --dataset COLLAB \
               --graph_pooling_type max --neighbor_pooling_type sum
```

Results
-------

Run with following with the double SUM pooling way:
(tested dataset: "MUTAG"(default), "COLLAB", "IMDBBINARY", "IMDBMULTI")
```bash
python3 train.py --dataset MUTAB --device 0  \
                --graph_pooling_type sum --neighbor_pooling_type sum
```

* MUTAG: 0.85 (paper: ~0.89)
* COLLAB: 0.89 (paper: ~0.80)
* IMDBBINARY: 0.76 (paper: ~0.75)
* IMDBMULTI: 0.51 (paper: ~0.52)
