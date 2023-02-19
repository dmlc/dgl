"""
.. _model-gcn:

Graph Convolutional Network
====================================

**Author:** `Qi Huang <https://github.com/HQ01>`_, `Minjie Wang  <https://jermainewang.github.io/>`_,
Yu Gai, Quan Gan, Zheng Zhang

.. warning::

    The tutorial aims at gaining insights into the paper, with code as a mean
    of explanation. The implementation thus is NOT optimized for running
    efficiency. For recommended implementation, please refer to the `official
    examples <https://github.com/dmlc/dgl/tree/master/examples>`_.

This is a gentle introduction of using DGL to implement Graph Convolutional
Networks (Kipf & Welling et al., `Semi-Supervised Classification with Graph
Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`_). We explain
what is under the hood of the :class:`~dgl.nn.GraphConv` module.
The reader is expected to learn how to define a new GNN layer using DGL's
message passing APIs.
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

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_u(u="h", out="m")
gcn_reduce = fn.sum(msg="m", out="h")

###############################################################################
# We then proceed to define the GCNLayer module. A GCNLayer essentially performs
# message passing on all the nodes then applies a fully-connected layer.
#
# .. note::
#
#    This is showing how to implement a GCN from scratch.  DGL provides a more
#    efficient :class:`builtin GCN layer module <dgl.nn.pytorch.conv.GraphConv>`.
#


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata["h"] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata["h"]
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

from dgl.data import CoraGraphDataset


def load_cora_data():
    dataset = CoraGraphDataset()
    g = dataset[0]
    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    test_mask = g.ndata["test_mask"]
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
# Add edges between each node and itself to preserve old node representations
g.add_edges(g.nodes(), g.nodes())
optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
dur = []
for epoch in range(50):
    if epoch >= 3:
        t0 = time.time()
    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)
    acc = evaluate(net, g, features, labels, test_mask)
    print(
        "Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)
        )
    )
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
# this layer. :math:`\tilde{D}` and :math:`\tilde{A}` are separately the degree
# and adjacency matrices for the graph. With the superscript ~, we are referring
# to the variant where we add additional edges between each node and itself to
# preserve its old representation in graph convolutions. The shape of the input
# :math:`H^{(0)}` is :math:`N \times D`, where :math:`N` is the number of nodes
# and :math:`D` is the number of input features. We can chain up multiple
# layers as such to produce a node-level representation output with shape
# :math:`N \times F`, where :math:`F` is the dimension of the output node
# feature vector.
#
# The equation can be efficiently implemented using sparse matrix
# multiplication kernels (such as Kipf's
# `pygcn <https://github.com/tkipf/pygcn>`_ code). The above DGL implementation
# in fact has already used this trick due to the use of builtin functions.
#
# Note that the tutorial code implements a simplified version of GCN where we
# replace :math:`\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}` with
# :math:`\tilde{A}`. For a full implementation, see our example
# `here  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcn>`_.
