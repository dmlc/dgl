# Deep Graph Library (DGL)
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=DGL/master)](http://ci.dgl.ai:80/job/DGL/job/master/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)

[Documentation](https://docs.dgl.ai) | [DGL at a glance](https://docs.dgl.ai/tutorials/basics/1_first.html#sphx-glr-tutorials-basics-1-first-py) |
[Model Tutorials](https://docs.dgl.ai/tutorials/models/index.html) | [Discussion Forum](https://discuss.dgl.ai)

Model Zoos: [Chemistry](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo) | [Citation Networks](https://github.com/dmlc/dgl/tree/master/examples/pytorch/model_zoo/citation_network)

DGL is a Python package that interfaces between existing tensor libraries and data being expressed as
graphs.

It makes implementing graph neural networks (including Graph Convolution Networks, TreeLSTM, and many others) easy while
maintaining high computation efficiency.

All model examples can be found [here](https://github.com/dmlc/dgl/tree/master/examples).

A summary of part of the model accuracy and training speed with the Pytorch backend (on Amazon EC2 p3.2x instance (w/ V100 GPU)), as compared with the best open-source implementations:

| Model                                                            | Reported <br> Accuracy | DGL <br> Accuracy | Author's training speed (epoch time)                                          | DGL speed (epoch time) | Improvement |
| -----                                                            | -----------------      | ------------      | ------------------------------------                                          | ---------------------- | ----------- |
| [GCN](https://arxiv.org/abs/1609.02907)                          | 81.5%                  | 81.0%             | [0.0051s (TF)](https://github.com/tkipf/gcn)                                  | 0.0031s                | 1.64x       |
| [GAT](https://arxiv.org/abs/1710.10903)                          | 83.0%                  | 83.9%             | [0.0982s (TF)](https://github.com/PetarV-/GAT)                                | 0.0113s                | 8.69x       |
| [SGC](https://arxiv.org/abs/1902.07153)                          | 81.0%                  | 81.9%             | n/a                                                                           | 0.0008s                | n/a         |
| [TreeLSTM](http://arxiv.org/abs/1503.00075)                      | 51.0%                  | 51.72%            | [14.02s (DyNet)](https://github.com/clab/dynet/tree/master/examples/treelstm) | 3.18s                  | 4.3x        |
| [R-GCN <br> (classification)](https://arxiv.org/abs/1703.06103)  | 73.23%                 | 73.53%            | [0.2853s (Theano)](https://github.com/tkipf/relational-gcn)                   | 0.0075s                | 38.2x       |
| [R-GCN <br> (link prediction)](https://arxiv.org/abs/1703.06103) | 0.158                  | 0.151             | [2.204s (TF)](https://github.com/MichSchli/RelationPrediction)                | 0.453s                 | 4.86x       |
| [JTNN](https://arxiv.org/abs/1802.04364)                         | 96.44%                 | 96.44%            | [1826s (Pytorch)](https://github.com/wengong-jin/icml18-jtnn)                 | 743s                   | 2.5x        |
| [LGNN](https://arxiv.org/abs/1705.08415)                         | 94%                    | 94%               | n/a                                                                           | 1.45s                  | n/a         |
| [DGMG](https://arxiv.org/pdf/1803.03324.pdf)                     | 84%                    | 90%               | n/a                                                                           | 238s                   | n/a         |

With the MXNet/Gluon backend , we scaled a graph of 50M nodes and 150M edges on a P3.8xlarge instance, 
with 160s per epoch, on SSE ([Stochastic Steady-state Embedding](https://www.cc.gatech.edu/~hdai8/pdf/equilibrium_embedding.pdf)), 
a model similar to GCN. 


We are currently in Beta stage.  More features and improvements are coming.

## News

v0.3 has just been released! Huge performance improvement (up to 19x). See release note
[here](https://github.com/dmlc/dgl/releases/tag/v0.3).

We presented DGL at [GTC 2019](https://www.nvidia.com/en-us/gtc/) as an
instructor-led training session. Check out our slides and tutorial materials
[here](https://github.com/dglai/DGL-GTC2019)!!!

## System requirements

DGL should work on

* all Linux distributions no earlier than Ubuntu 16.04
* macOS X
* Windows 10

DGL also requires Python 3.5 or later.  Python 2 support is coming.

Right now, DGL works on [PyTorch](https://pytorch.org) 0.4.1+ and [MXNet](https://mxnet.apache.org) nightly
build.

## Installation

### Using anaconda

```
conda install -c dglteam dgl           # cpu version
conda install -c dglteam dgl-cuda9.0   # CUDA 9.0
conda install -c dglteam dgl-cuda9.2   # CUDA 9.2
conda install -c dglteam dgl-cuda10.0  # CUDA 10.0
```

### Using pip

```
pip install dgl       # cpu version
pip install dgl-cu90  # CUDA 9.0
pip install dgl-cu92  # CUDA 9.2
pip install dgl-cu100 # CUDA 10.0
```

### From source

Refer to the guide [here](https://docs.dgl.ai/install/index.html#install-from-source).

## How DGL looks like

A graph can be constructed with feature tensors like this:

```python
import dgl
import torch as th

g = dgl.DGLGraph()
g.add_nodes(5)                          # add 5 nodes
g.add_edges([0, 0, 0, 0], [1, 2, 3, 4]) # add 4 edges 0->1, 0->2, 0->3, 0->4
g.ndata['h'] = th.randn(5, 3)           # assign one 3D vector to each node
g.edata['h'] = th.randn(4, 4)           # assign one 4D vector to each edge
```

This is *everything* to implement a single layer for Graph Convolutional Network on PyTorch:

```python
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

msg_func = fn.copy_src(src='h', out='m')
reduce_func = fn.sum(msg='m', out='h')

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def apply(self, nodes):
        return {'h': F.relu(self.linear(nodes.data['h']))}

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(msg_func, reduce_func)
        g.apply_nodes(func=self.apply)
        return g.ndata.pop('h')
```

One can also customize how message and reduce function works.  The following code
demonstrates a (simplified version of) Graph Attention Network (GAT) layer:

```python
def msg_func(edges):
    return {'k': edges.src['k'], 'v': edges.src['v']}

def reduce_func(nodes):
    # nodes.data['q'] has the shape
    #     (number_of_nodes, feature_dims)
    # nodes.data['k'] and nodes.data['v'] have the shape
    #     (number_of_nodes, number_of_incoming_messages, feature_dims)
    # You only need to deal with the case where all nodes have the same number
    # of incoming messages.
    q = nodes.data['q'][:, None]
    k = nodes.mailbox['k']
    v = nodes.mailbox['v']
    s = F.softmax((q * k).sum(-1), 1)[:, :, None]
    return {'v': th.sum(s * v, 1)}

class GATLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GATLayer, self).__init__()
        self.Q = nn.Linear(in_feats, out_feats)
        self.K = nn.Linear(in_feats, out_feats)
        self.V = nn.Linear(in_feats, out_feats)

    def apply(self, nodes):
        return {'v': F.relu(self.linear(nodes.data['v']))}

    def forward(self, g, feature):
        g.ndata['v'] = self.V(feature)
        g.ndata['q'] = self.Q(feature)
        g.ndata['k'] = self.K(feature)
        g.update_all(msg_func, reduce_func)
        g.apply_nodes(func=self.apply)
        return g.ndata['v']
```

For the basics of coding with DGL, please see [DGL basics](https://docs.dgl.ai/tutorials/basics/index.html).

For more realistic, end-to-end examples, please see [model tutorials](https://docs.dgl.ai/tutorials/models/index.html).


## New to Deep Learning?

Check out the open source book [*Dive into Deep Learning*](http://gluon.ai/).


## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/dmlc/dgl/issues).

We welcome all contributions from bug fixes to new features and extensions.
We expect all contributions discussed in the issue tracker and going through PRs.  Please refer to our [contribution guide](https://docs.dgl.ai/contribute.html).

## Cite

If you use DGL in a scientific publication, we would appreciate citations to the following paper:
```
@article{wang2019dgl,
    title={Deep Graph Library: Towards Efficient and Scalable Deep Learning on Graphs},
    url={https://arxiv.org/abs/1909.01315},
    author={{Wang, Minjie and Yu, Lingfan and Zheng, Da and Gan, Quan and Gai, Yu and Ye, Zihao and Li, Mufei and Zhou, Jinjing and Huang, Qi and Ma, Chao and Huang, Ziyue and Guo, Qipeng and Zhang, Hao and Lin, Haibin and Zhao, Junbo and Li, Jinyang and Smola, Alexander J and Zhang, Zheng},
    journal={ICLR Workshop on Representation Learning on Graphs and Manifolds},
    year={2019}
}
```

## The Team

DGL is developed and maintained by [NYU, NYU Shanghai, AWS Shanghai AI Lab, and AWS MXNet Science Team](https://www.dgl.ai/pages/about.html).


## License

DGL uses Apache License 2.0.
