"""
.. currentmodule:: dgl

Batched Graph Classification with DGL
=====================================

**Author**: `Mufei Li <https://github.com/mufeili>`_,
`Minjie Wang <https://jermainewang.github.io/>`_

Graph classification, namely the prediction of graph labels, is an important problem
with applications across many fields -- bioinformatics, chemoinformatics, social
network analysis, urban computing and cyber-security. Recently there has been an
arising trend of applying graph neural networks to graph classification (
`Ying et al., 2018 <https://arxiv.org/pdf/1806.08804.pdf>`_,
`Cangea et al., 2018 <https://arxiv.org/pdf/1811.01287.pdf>`_,
`Knyazev et al., 2018 <https://arxiv.org/pdf/1811.09595.pdf>`_,
`Bianchi et al., 2019 <https://arxiv.org/pdf/1901.01343.pdf>`_,
`Liao et al., 2019 <https://arxiv.org/pdf/1901.01484.pdf>`_,
`Gao et al., 2019 <https://openreview.net/pdf?id=HJePRoAct7>`_).

This tutorial is a demonstration for
 * batching multiple graphs of variable size and shape with DGL
 * training a graph neural network for a simple graph classification task
"""

###############################################################################
# Simple Graph Classification Task
# --------------------------------
# In this tutorial, we will learn how to perform batched graph classification
# with dgl via a toy example of distinguishing cycles from stars. From a synthetic
# dataset, we want to learn a binary classifier like below:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/classifier.png
#     :width: 400pt
#     :align: center
#
# Dataset
# -------
# We implement a dataset class for our synthetic dataset as usual in PyTorch.
# This will be a balanced dataset with half of cycles and half of stars. In
# particular for the ``collate`` function, we perform a ``dgl.batch`` operation.
# Normally in ``collate`` function we batch across a set of tensors with same
# shape to make a minibatch, here our ``dgl.batch`` performs a similar operation
# across graphs except that our graphs may have different sizes. Below is a
# visualization that hopefully gives a general idea:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/batch.png
#     :width: 400pt
#     :align: center
#
# Basically with ``dgl.batch([g_1,...,g_n])``, we can merge :math:`n` small
# graphs into a large graph with :math:`n` connected components. This allows
# many possibilities of parallelization. To learn more, check
# :class:`BatchedDGLGraph`.

import dgl
import networkx as nx
import random
from torch.utils.data import Dataset, DataLoader

def collate(samples):
    # Convert a list of tuples to two lists.
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################
# Graph Classifier
# ----------------
# The graph classification can be proceeded as follows:
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/graph_classifier.png
#
# From a batch of graphs, we first perform message passing/graph convolution
# for nodes to "communicate" with others. After message passing, we compute a
# tensor for graph representation from node (and edge) attributes. This step may
# be called "readout/aggregation" interchangeably. Finally, the graph
# representations can be fed into a classifier :math:`g` for classification.
#
# Graph Convolution
# -----------------
# Here we employ the rule below for message passing and update node features:
#
# .. math::
#
#    h_{v}^{(l+1)} = \text{ReLU}\left(b^{(l)}+\frac{1}{|\mathcal{N}(v)|+1}\sum_{u\in\mathcal{N}(v)\bigcup\{v\}}h_{u}^{(l)}W^{(l)}\right),
#
# where :math:`h_{v}^{(0)}` is the initial feature of node :math:`v`,
# :math:`\mathcal{N}(v)` is the collection of neighbors of node :math:`v`,
# :math:`b^{(l)}` and :math:`W^{(l)}` are trainable parameters.
#
# Our main class for this stage is ``GraphConvolution``, where simultaneously
# each node :math:`v` sends a message of its feature :math:`h_{v}^{(l)}` to
# all their neighbors with ``msg``, and then based on the messages received,
# each node performs
# :math:`\frac{1}{|\mathcal{N}(v)|+1}\sum_{u\in\mathcal{N}(v)\bigcup\{v\}}h_{u}^{(l)}`
# in ``reduce``. Finally ``update_func`` takes in
# :math:`\frac{1}{|\mathcal{N}(v)|+1}\sum_{u\in\mathcal{N}(v)\bigcup\{v\}}h_{u}^{(l)}`
# and returns :math:`h_{v}^{(l+1)}`.
#
# Note that the self edges added in the dataset initialization allows us to
# include the original node feature :math:`h_{v}^{(l)}` when taking the average.

import torch
import torch.nn as nn


def msg(edges):
    """Sends a message of node feature hv."""
    msg = edges.src['h']
    return {'m': msg}

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeUpdate(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, node_field, in_dim, out_dim):
        super(NodeUpdate, self).__init__()

        self.node_field = node_field
        self.update_func = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.ReLU())

    def forward(self, node):
        return {self.node_field:
                    self.update_func(node.data[self.node_field])}

class GraphConvolution(nn.Module):
    def __init__(self, node_field, in_dim, out_dim):
        super(GraphConvolution, self).__init__()

        self.node_field = node_field
        self.update_func = NodeUpdate(node_field, in_dim, out_dim)

    def forward(self, g, h):
        # Initialize the node features with h.
        g.ndata[self.node_field] = h
        g.update_all(msg, reduce, self.update_func)
        h = g.ndata.pop(self.node_field)
        return h

###############################################################################
# Readout and Classification
# --------------------------
# For this demonstration, we consider initial node features to be their degrees.
# After two rounds of graph convolution, we perform a graph readout by averaging
# over all node features :math:`\frac{1}{|\mathcal{V}|}\sum_{v\in\mathcal{V}}h_{v}`
# for each graph in the batch. ``dgl.mean(g)`` handles this task for a batch of
# graphs with variable size. We then feed our graph representations into a
# classifier with one linear layer followed by :math:`\text{sigmoid}`.

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GraphConvolution('h', in_dim, hidden_dim),
            GraphConvolution('h', hidden_dim, hidden_dim)])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        graph_repr = dgl.mean_nodes(g, 'h')
        return self.classify(graph_repr)

###############################################################################
# Setup and Training
# ------------------
# We create a synthetic dataset of :math:`1000` graphs with :math:`10` ~
# :math:`20` nodes.

import torch.optim as optim
from dgl.data import MiniGCDataset

# We use a 80:20 split for the training set and the test set.
trainset = MiniGCDataset(1000, 10, 20)
testset = MiniGCDataset(200, 10, 20)
data_loader = DataLoader(trainset, batch_size=32, shuffle=True,
                         collate_fn=collate)
model = Classifier(1, 32, 8)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()

epoch_losses = []
for epoch in range(50):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(data_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    print('Epoch {}, loss {:.4f}'.format(epoch, loss))
    epoch_losses.append(epoch_loss / (iter + 1))

###############################################################################
# The learning curve of a run is presented below:

import matplotlib.pyplot as plt

plt.title('cross entropy averaged over minibatches')
plt.plot(epoch_losses)
plt.show()

###############################################################################
# On our test set with the trained classifier, the accuracy of sampled
# predictions varies across :math:`10` random runs between :math:`90.5` % ~
# :math:`100` % with the code below:

# Convert a list of tuples to two lists
model.eval()
test_X, test_Y = map(list, zip(*testset))
test_bg = dgl.batch(test_X)
test_Y = torch.tensor(test_Y).float().view(-1, 1)
sampled_Y = torch.multinomial(torch.softmax(model(test_bg), 1), 1)
print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
    (test_Y == sampled_Y.float()).sum().item() / len(test_Y) * 100))

###############################################################################
# We also created an animation for the soft classification performed on the
# test set by one model we trained. Recall that we have label :math:`0` for
# cycle graph and label :math:`1` for star graph.
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/batch/test_eval.gif
#
# What's Next?
# ------------
# Graph classification with graph neural networks is still a very young field
# waiting for folks to bring more exciting discoveries! It is not easy as it
# requires mapping different graphs to different embeddings while preserving
# their structural similarity in the embedding space. To learn more about it,
# `"How Powerful Are Graph Neural Networks?" <https://arxiv.org/pdf/1810.00826.pdf>`_
# in ICLR 2019 might be a good starting point.
#
# With regards to more examples on batched graph processing, see
#
# * our tutorials on `Tree LSTM <https://docs.dgl.ai/tutorials/models/2_small_graph/3_tree-lstm.html>`_ and `Deep Generative Models of Graphs <https://docs.dgl.ai/tutorials/models/3_generative_model/5_dgmg.html>`_
# * an example implementation of `Junction Tree VAE <https://github.com/dmlc/dgl/tree/master/examples/pytorch/jtnn>`_
