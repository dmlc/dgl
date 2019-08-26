Graph Attention Networks (GAT)
============

- Paper link: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)
- Author's code repo:
  [https://github.com/PetarV-/GAT](https://github.com/PetarV-/GAT).

Note that the original code is implemented with Tensorflow for the paper.

### Dependencies
* MXNet nightly build
* requests

```bash
pip install mxnet --pre
pip install requests
```


### Usage (make sure that DGLBACKEND is changed into mxnet)
```bash
DGLBACKEND=mxnet python3 train.py --dataset cora --gpu 0 --num-heads 8
```
