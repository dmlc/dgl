"""
.. _model-gcn:

Graph Convolutional Network
====================================

**Author:** `Qi Huang <https://github.com/HQ01>`_, `Minjie Wang  <https://jermainewang.github.io/>`_,
Yu Gai, Quan Gan, Zheng Zhang

This is a gentle introduction of using DGL to implement Graph Convolutional
Networks (Kipf & Welling et al., `Semi-Supervised Classification with Graph
Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`_). We build upon
the :doc:`earlier tutorial <../../basics/3_pagerank>` on DGLGraph and demonstrate
how DGL combines graph with deep neural network and learn structural representations.
"""

###############################################################################
# Model Overview
# ------------------------------------------
# GCN from the perspective of message passing
# ```````````````````````````````````````````````
# We describe a layer of graph convolutional neural network from a message
# passing perspective; the math can be found `here <math_>`_.
# It boils down to the following step, for each node :math:`u`:
# 
# 1) Aggregate neighbors' representations :math:`h_{v}` to produce an
# intermediate representation :math:`\hat{h}_u`.  2) Transform the aggregated
# representation :math:`\hat{h}_{u}` with a linear projection followed by a
# non-linearity: :math:`h_{u} = f(W_{u} \hat{h}_u)`.
# 
# We will implement step 1 with DGL message passing, and step 2 with the
# ``apply_nodes`` method, whose node UDF will be a PyTorch ``nn.Module``.
# 
# GCN implementation with DGL
# ``````````````````````````````````````````
# We first define the message and reduce function as usual.  Since the
# aggregation on a node :math:`u` only involves summing over the neighbors'
# representations :math:`h_v`, we can simply use builtin functions:

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

###############################################################################
# We then define the node UDF for ``apply_nodes``, which is a fully-connected layer:

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

###############################################################################
# We then proceed to define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. Note
# that we omitted the dropout in the paper for simplicity.

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7).

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)
    
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print(net)

###############################################################################
# We load the cora dataset using DGL's built-in data module.

from dgl.data import citation_graph as citegrh
import networkx as nx
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, mask

###############################################################################
# We then train the network as follows:

import time
import numpy as np
g, features, labels, mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(30):
    if epoch >=3:
        t0 = time.time()
        
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)
    
    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))

###############################################################################
# .. _math:
#
# GCN in one formula
# ------------------
# Mathematically, the GCN model follows this formula:
# 
# :math:`H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})`
# 
# Here, :math:`H^{(l)}` denotes the :math:`l^{th}` layer in the network,
# :math:`\sigma` is the non-linearity, and :math:`W` is the weight matrix for
# this layer. :math:`D` and :math:`A`, as commonly seen, represent degree
# matrix and adjacency matrix, respectively. The ~ is a renormalization trick
# in which we add a self-connection to each node of the graph, and build the
# corresponding degree and adjacency matrix.  The shape of the input
# :math:`H^{(0)}` is :math:`N \times D`, where :math:`N` is the number of nodes
# and :math:`D` is the number of input features. We can chain up multiple
# layers as such to produce a node-level representation output with shape
# :math`N \times F`, where :math:`F` is the dimension of the output node
# feature vector.
# 
# The equation can be efficiently implemented using sparse matrix
# multiplication kernels (such as Kipf's
# `pygcn <https://github.com/tkipf/pygcn>`_ code). The above DGL implementation
# in fact has already used this trick due to the use of builtin functions. To
# understand what is under the hood, please read our tutorial on :doc:`PageRank <../../basics/3_pagerank>`.
