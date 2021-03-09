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
python train.py --dataset ENZYMES --pool_ratio 0.10 --num_pool 1 --epochs 1000
python train.py --dataset DD --pool_ratio 0.15 --num_pool 1  --batch-size 10
```
Performance
-----------
ENZYMES 63.33% (with early stopping)
DD 79.31% (with early stopping)


## Update (2021-03-09)

**Changes:**

* Fix bug in Diffpool: the wrong `assign_dim` parameter
* Improve efficiency of DiffPool, make the model independent of batch size. Remove redundant computation.


**Efficiency:**

On V100-SXM2 16GB

| Experiments       | Train time per epoch(s) | Memory usage(MB) |
| ----------------- | ----------------------: | ---------------: |
| Original-DD-bs=20 | OOM                     | OOM              |
| Original-DD-bs=10 | 21.302                  | 5274.620         |
| Original-ENZYMES  | 1.749                   | 25.685           |
| Improved-DD-bs=20 | 44.682                  | 10088.889        |
| Improved-DD-bs=10 | 17.282                  | 2928.568         |
| Improved-ENZYMES  | 1.685                   | 21.909           |

**Accuracy**

Each experiment with improved model is only conducted once, thus the result may has noise.

| Experiments      | Acc    |
| ---------------- | -----: |
| Original-DD      | 79.31% |
| Original-ENZYMES | 63.33% |
| Improved-DD      | 78.33% |
| Improved-ENZYMES | 68.33% |
