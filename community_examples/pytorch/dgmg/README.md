# Learning Deep Generative Models of Graphs

This is an implementation of [Learning Deep Generative Models of Graphs](https://arxiv.org/pdf/1803.03324.pdf) by 
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia.

For molecule generation, see 
[DGL-LifeSci](https://github.com/awslabs/dgl-lifesci/tree/master/examples/generative_models/dgmg).

## Dependencies
- Python 3.5.2
- [Pytorch 0.4.1](https://pytorch.org/)
- [Matplotlib 2.2.2](https://matplotlib.org/)

## Usage

`python3 main.py`

## Performance

90% accuracy for cycles compared with 84% accuracy reported in the original paper.

## Speed

On AWS p3.2x instance (w/ V100), one epoch takes ~526s.

## Acknowledgement

We would like to thank Yujia Li for providing details on the implementation.
