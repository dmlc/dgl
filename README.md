# DGL
[![Build Status](http://ci.dgl.ai:80/buildStatus/icon?job=DGL/master)](http://ci.dgl.ai:80/job/DGL/job/master/)
[![GitHub license](https://dmlc.github.io/img/apache2.svg)](./LICENSE)

DGL is a Python package that interfaces between existing tensor libraries and data being expressed as
graphs.

It makes implementing graph neural networks (including Graph Convolution Networks, TreeLSTM, and many others) easy while
maintaining high computation efficiency.

A summary of the model accuracy and training speed with the Pytorch backend (on Amazon EC2 p3.2x instance (w/ V100 GPU)).

| Model | Reported <br> Accuracy | DGL <br> Accuracy | Author's training speed (epoch time) | DGL speed (epoch time) | Improvement |
| ----- | ----------------- | ------------ | ------------------------------------ | ---------------------- | ----------- |
| [GCN]()   | [81.5%]() | 81.0% | 0.0051s (TF) | 0.0042s | 1.17x |
| TreeLSTM | 51.0% | 51.72% | 14.02s (DyNet) | 3.18s | 4.3x |
| R-GCN <br> (classification) | 73.23% | 73.53% | 0.2853s (Theano) | 0.0273s | 10.4x |
| R-GCN <br> (link prediction) | 0.158 | 0.151 | 2.204s (TF) | 0.633s | 3.5x |
| JTNN | 96.44% | 96.44% | 1826s (Pytorch) | 743s | 2.5x |
| LGNN | 94% | 94% | n/a | 1.45s | n/a |
| DGMG | 84% | 90% | n/a | 1 hr | n/a |

For scalability, with the MXNet backend (on a P3.8xlarge instance), we have successfully scaled [Stochastic Steady-state Embedding (SSE)](https://www.cc.gatech.edu/~hdai8/pdf/equilibrium_embedding.pdf), a model similar to Graph convolution network (GCN) to a graph with 50 million nodes and 150 million edges. One epoch only takes about 160 seconds.

We are currently in Beta stage.  More features and improvements are coming.

## System requirements

DGL should work on

* all Linux distributions no earlier than Ubuntu 16.04
* macOS X
* Windows 7 or later

DGL also requires Python 3.5 or later.  Python 2 support is coming.

Right now, DGL works on [PyTorch](https://pytorch.org) 0.4.1+ and [MXNet](mxnet.apache.org) nightly
build.

## Installation

### Using anaconda

```
conda install -c dglteam dgl
```

### Using pip

```
pip install dgl
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

## Contributing

Please let us know if you encounter a bug or have any suggestions by [filing an issue](https://github.com/jermainewang/dgl/issues).

We welcome all contributions from bug fixes to new features and extensions.
We expect all contributions discussed in the issue tracker and going through PRs.  Please refer to the PR guide.

## The Team

DGL is developed and maintained by [NYU, NYU Shanghai, AWS Shanghai AI Lab, and AWS MXNet Science Team](https://www.dgl.ai/about).

## License

DGL uses Apache License 2.0.
