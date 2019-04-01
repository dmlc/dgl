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
python main.py
```

An experiment on the GIN in customized settings can be run with
```bash
python main.py --device 0 --disable_cuda --dataset COLLAB \
               --graph_pooling_type max --neighbor_pooling_type sum
```
