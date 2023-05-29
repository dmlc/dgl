"""
Node Classification with DGL
============================

GNNs are powerful tools for many machine learning tasks on graphs. In
this introductory tutorial, you will learn the basic workflow of using
GNNs for node classification, i.e. predicting the category of a node in
a graph.

By completing this tutorial, you will be able to

-  Load a DGL-provided dataset.
-  Build a GNN model with DGL-provided neural network modules.
-  Train and evaluate a GNN model for node classification on either CPU
   or GPU.

This tutorial assumes that you have experience in building neural
networks with PyTorch.

(Time estimate: 13 minutes)

"""

import os

os.environ["DGLBACKEND"] = "pytorch"
import dgl
import dgl.data
import torch
import torch.nn as nn
import torch.nn.functional as F

######################################################################
# Overview of Node Classification with GNN
# ----------------------------------------
#
# One of the most popular and widely adopted tasks on graph data is node
# classification, where a model needs to predict the ground truth category
# of each node. Before graph neural networks, many proposed methods are
# using either connectivity alone (such as DeepWalk or node2vec), or simple
# combinations of connectivity and the node's own features.  GNNs, by
# contrast, offers an opportunity to obtain node representations by
# combining the connectivity and features of a *local neighborhood*.
#
# `Kipf et
# al., <https://arxiv.org/abs/1609.02907>`__ is an example that formulates
# the node classification problem as a semi-supervised node classification
# task. With the help of only a small portion of labeled nodes, a graph
# neural network (GNN) can accurately predict the node category of the
# others.
#
# This tutorial will show how to build such a GNN for semi-supervised node
# classification with only a small number of labels on the Cora
# dataset,
# a citation network with papers as nodes and citations as edges. The task
# is to predict the category of a given paper. Each paper node contains a
# word count vector as its features, normalized so that they sum up to one,
# as described in Section 5.2 of
# `the paper <https://arxiv.org/abs/1609.02907>`__.
#
# Loading Cora Dataset
# --------------------
#


dataset = dgl.data.CoraGraphDataset()
print(f"Number of categories: {dataset.num_classes}")


######################################################################
# A DGL Dataset object may contain one or multiple graphs. The Cora
# dataset used in this tutorial only consists of one single graph.
#

g = dataset[0]


######################################################################
# A DGL graph can store node features and edge features in two
# dictionary-like attributes called ``ndata`` and ``edata``.
# In the DGL Cora dataset, the graph contains the following node features:
#
# - ``train_mask``: A boolean tensor indicating whether the node is in the
#   training set.
#
# - ``val_mask``: A boolean tensor indicating whether the node is in the
#   validation set.
#
# - ``test_mask``: A boolean tensor indicating whether the node is in the
#   test set.
#
# - ``label``: The ground truth node category.
#
# -  ``feat``: The node features.
#

print("Node features")
print(g.ndata)
print("Edge features")
print(g.edata)


######################################################################
# Defining a Graph Convolutional Network (GCN)
# --------------------------------------------
#
# This tutorial will build a two-layer `Graph Convolutional Network
# (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each
# layer computes new node representations by aggregating neighbor
# information.
#
# To build a multi-layer GCN you can simply stack ``dgl.nn.GraphConv``
# modules, which inherit ``torch.nn.Module``.
#

from dgl.nn import GraphConv


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


# Create the model with given dimensions
model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)


######################################################################
# DGL provides implementation of many popular neighbor aggregation
# modules. You can easily invoke them with one line of code.
#


######################################################################
# Training the GCN
# ----------------
#
# Training this GCN is similar to training other PyTorch neural networks.
#


def train(g, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    best_val_acc = 0
    best_test_acc = 0

    features = g.ndata["feat"]
    labels = g.ndata["label"]
    train_mask = g.ndata["train_mask"]
    val_mask = g.ndata["val_mask"]
    test_mask = g.ndata["test_mask"]
    for e in range(100):
        # Forward
        logits = model(g, features)

        # Compute prediction
        pred = logits.argmax(1)

        # Compute loss
        # Note that you should only compute the losses of the nodes in the training set.
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Compute accuracy on training/validation/test
        train_acc = (pred[train_mask] == labels[train_mask]).float().mean()
        val_acc = (pred[val_mask] == labels[val_mask]).float().mean()
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()

        # Save the best validation accuracy and the corresponding test accuracy.
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if e % 5 == 0:
            print(
                f"In epoch {e}, loss: {loss:.3f}, val acc: {val_acc:.3f} (best {best_val_acc:.3f}), test acc: {test_acc:.3f} (best {best_test_acc:.3f})"
            )


model = GCN(g.ndata["feat"].shape[1], 16, dataset.num_classes)
train(g, model)


######################################################################
# Training on GPU
# ---------------
#
# Training on GPU requires to put both the model and the graph onto GPU
# with the ``to`` method, similar to what you will do in PyTorch.
#
# .. code:: python
#
#    g = g.to('cuda')
#    model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes).to('cuda')
#    train(g, model)
#


######################################################################
# What’s next?
# ------------
#
# -  :doc:`How does DGL represent a graph <2_dglgraph>`?
# -  :doc:`Write your own GNN module <3_message_passing>`.
# -  :doc:`Link prediction (predicting existence of edges) on full
#    graph <4_link_predict>`.
# -  :doc:`Graph classification <5_graph_classification>`.
# -  :doc:`Make your own dataset <6_load_data>`.
# -  :ref:`The list of supported graph convolution
#    modules <apinn-pytorch>`.
# -  :ref:`The list of datasets provided by DGL <apidata>`.
#


# Thumbnail credits: Stanford CS224W Notes
# sphinx_gallery_thumbnail_path = '_static/blitz_1_introduction.png'
