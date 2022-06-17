Geometric Deep Learning models
=========

This example shows how to use geometric deep learning models defined in `dgl.nn.pytorch.conv` for
graph classification.

Currently we support following models:
- [ChebNet](https://arxiv.org/pdf/1606.09375.pdf)
- [MoNet](https://arxiv.org/pdf/1611.08402.pdf)

## Image Classification on MNIST

By transforming images to graphs, graph classifcation algorithms could
be applied to image classification problems.

### Usage
```bash
python mnist.py --model cheb --gpu 0
python mnist.py --model monet --gpu 0
```

### Acknowledgement
We thank [Xavier Bresson](https://github.com/xbresson) for providing 
code for graph coarsening algorithm and grid graph building in  
[CE7454_2019 Labs](https://github.com/xbresson/CE7454_2019/tree/master/codes/labs_lecture14/lab01_ChebGCNs).
