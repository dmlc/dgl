# Learning Deep Generative Models of Graphs

This is an implementation of [Learning Deep Generative Models of Graphs](https://arxiv.org/pdf/1803.03324.pdf) by 
Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia. In particular, we use:
- DL framework: PyTorch 0.4.1
- Graph neural network framework: DGL 0.5.dev with commit `cab1fdf`

# Dependency
- Python 3.6.7
- [Pytorch 0.4.1](https://pytorch.org/)
- [DGL 0.5.dev](https://github.com/jermainewang/dgl)
- [Networkx 2.1](https://networkx.github.io/): "for the creation, manipulation, and study of the structure, dynamics, 
and functions of complex networks".

# Usage

- Train with batch size 1: `python main.py`
