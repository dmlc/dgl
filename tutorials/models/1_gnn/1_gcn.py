"""
.. _model-gcn:

Graph Convolutional Network
====================================

**Author:** `Qi Huang <https://github.com/HQ01>`_, `Minjie Wang  <https://jermainewang.github.io/>`_,
Yu Gai, Quan Gan, Zheng Zhang

This is a gentle introduction of using DGL to implement Graph Convolutional
Networks (Kipf & Welling et al., `Semi-Supervised Classification with Graph
Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`_). We explain
what is under the hood of the :class:`~dgl.nn.pytorch.GraphConv` module.
The reader is expected to learn how to define a new GNN layer using DGL's
message passing APIs.

We build upon the :doc:`earlier tutorial <../../basics/3_pagerank>` on DGLGraph
and demonstrate how DGL combines graph with deep neural network and learn
structural representations.
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
# We will implement step 1 with DGL message passing, and step 2 by
# PyTorch ``nn.Module``.
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
# We then proceed to define the GCNLayer module. A GCNLayer essentially performs
# message passing on all the nodes then applies a fully-connected layer.

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7). The last GCN layer computes node embeddings,
# so the last layer in general does not apply activation.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(1433, 16)
        self.layer2 = GCNLayer(16, 7)
    
    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
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
    train_mask = th.BoolTensor(data.train_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = DGLGraph(data.graph)
    return g, features, labels, train_mask, test_mask

###############################################################################
# When a model is trained, we can use the following method to evaluate
# the performance of the model on the test dataset:

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

###############################################################################
# We then train the network as follows:

import time
import numpy as np
g, features, labels, train_mask, test_mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)
    
    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

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
