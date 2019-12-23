# Learning Deep Generative Models of Graphs

This is an implementation of [Learning Deep Generative Models of Graphs](https://arxiv.org/pdf/1803.03324.pdf) by 
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia.

For molecule generation, see 
[our model zoo for Chemistry](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/chem/generative_models/dgmg).

## Dependencies
- Python 3.5.2
- [Pytorch 0.4.1](https://pytorch.org/)
- [Matplotlib 2.2.2](https://matplotlib.org/)

## Usage

- Train with batch size 1: `python3 main.py`
- Train with batch size larger than 1: `python3 main_batch.py`.

## Performance

90% accuracy for cycles compared with 84% accuracy reported in the original paper.

## Speed

On AWS p3.2x instance (w/ V100), one epoch takes ~526s for batch size 1 and takes
~238s for batch size 10.

## Acknowledgement

We would like to thank Yujia Li for providing details on the implementation.
