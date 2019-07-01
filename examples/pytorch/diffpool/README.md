Hierarchical Graph Representation Learning with Differentiable Pooling
============


Paper link: [https://arxiv.org/abs/1806.08804](https://arxiv.org/abs/1806.08804)

Author's code repo: [https://github.com/RexYing/diffpool](https://github.com/RexYing/diffpool)

This folder contains a DGL implementation of the DiffPool model. The first pooling layer is computed with DGL, and following pooling layers are computed with tensorized operation since the pooled graphs are dense.

Dependencies
------------
* PyTorch 1.0+

How to run
----------

```bash
python train.py --dataset ENZYMES --pool_ratio 0.10 --num_pool 1
python train.py --dataset DD --pool_ratio 0.15 --num_pool 1  
```
Performance
-----------
ENZYMES 63.33% (with early stopping)
DD 79.31% (with early stopping)


## Dependencies

