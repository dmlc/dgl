"""
Training a GNN for Graph Classification
=======================================

By the end of this tutorial, you will be able to

-  Load a DGL-provided graph classification dataset.
-  Understand what readout function does.
-  Understand how to deal with a minibatch of graphs.
-  Build a GNN-based graph classification model.
-  Train and evaluate the model on a DGL-provided dataset.

"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


######################################################################
# Overview of Graph Classification with GNN
# -----------------------------------------
# 
# Graph classification or regression requires a model to predict certain
# graph-level properties of a single graph given its node and edge
# features. Common cases include molecular property prediction, etc.
# 
# This tutorial shows how to train a graph classification model for a
# small dataset from the paper `How Powerful Are Graph Neural
# Networks <https://arxiv.org/abs/1810.00826>`__.
# 
# Loading Data
# ------------
# 

import dgl.data

# Generate a synthetic dataset with 10000 graphs, ranging from 10 to 500 nodes.
dataset = dgl.data.GINDataset('PROTEINS', self_loop=True)


######################################################################
# The dataset is a set of graphs, each with node features and a single
# label. One can see the node feature dimensionality and the number of
# possible graph categories of ``GINDataset`` objects in ``dim_nfeats``
# and ``gclasses`` attributes.
# 

print('Node feature dimensionality:', dataset.dim_nfeats)
print('Number of graph categories:', dataset.gclasses)


######################################################################
# Define Data Loader
# ------------------
# 
# When you train a model for image classification or language modeling,
# you will use a ``DataLoader`` to iterate over the dataset. In DGL, you
# can still use the ``DataLoader``. The only thing you need to do is to
# write a ``collate`` function for the ``DataLoader`` that combines the
# examples together into a minibatch.
# 
# Combining the examples involves combining the graphs into a single
# bigger batched graph with ``dgl.batch()``, and concatenating the labels.
# ``dgl.batch()`` not only combines a list of graphs into a bigger graph
# with each element as a separate component, but also concatenates the
# node and edge features of the original graphs.
# 

from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

def collate(examples):
    graphs, labels = zip(*examples)
    return dgl.batch(graphs), torch.stack(labels)


######################################################################
# You can also use various dataset samplers provided in
# |torch.utils.data.sampler|_.
# For example, this tutorial creates a training ``DataLoader`` and test
# ``DataLoader``, using ``SubsetRandomSampler`` to tell PyTorch to sample
# from only a subset of the dataset.
#
# .. |torch.utils.data.sampler| replace:: ``torch.utils.data.sampler``
# .. _torch.utils.data.sampler: https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler
# 

num_examples = len(dataset)
num_train = int(num_examples * 0.8)

train_sampler = SubsetRandomSampler(torch.arange(num_train))
test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

train_dataloader = DataLoader(
    dataset, sampler=train_sampler, batch_size=5, collate_fn=collate, drop_last=False)
test_dataloader = DataLoader(
    dataset, sampler=test_sampler, batch_size=5, collate_fn=collate, drop_last=False)


######################################################################
# You can try to iterate over the created ``DataLoader`` and see what it
# gives:
# 

it = iter(train_dataloader)
batch = next(it)
print(batch)


######################################################################
# As you defined in the ``collate`` function, the ``DataLoader`` will
# return two objects for each iteration. Th second element is simply a
# label vector representing the category of each graph in the minibatch.
# The first element will be described below.
# 


######################################################################
# What is a batched graph?
# ------------------------
# 
# The first element is a batched ``DGLGraph``, which itself is also a
# ``DGLGraph`` (so you can still treat it as a normal ``DGLGraph`` object
# as in :doc:`here <2_dglgraph>`). It however contains the information
# necessary for recovering the original graphs, such as the number of
# nodes and edges of each graph element.
# 

batched_graph, labels = batch
print('Number of nodes for each graph element in the batch:', batched_graph.batch_num_nodes())
print('Number of edges for each graph element in the batch:', batched_graph.batch_num_edges())

# Recover the original graph elements from the minibatch
graphs = dgl.unbatch(batched_graph)
print('The original graphs in the minibatch:')
print(graphs)


######################################################################
# Define Model
# ------------
# 
# This tutorial will build a two-layer `Graph Convolutional Network
# (GCN) <http://tkipf.github.io/graph-convolutional-networks/>`__. Each of
# its layer computes new node representations by aggregating neighbor
# information. If you have gone through the
# :doc:`introduction <1_introduction>`, you will need to notice two
# differences:
# 
# -  Since you will be predicting a single category for the *entire graph*
#    instead of for every node, you will need to aggregate the
#    representations of all the nodes and potentially the edges to form a
#    graph-level representation. Such process is more commonly referred as
#    a *readout*. A simple choice is to average the node features of a
#    graph with ``dgl.mean_nodes()``.
#
# -  The input graph to the model will be a batched graph yielded by the
#    ``DataLoader``. The readout functions provided by DGL can handle
#    batched graphs so that they will return one representation for each
#    minibatch element.
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
        g.ndata['h'] = h
        return dgl.mean_nodes(g, 'h')



######################################################################
# Training Loop
# -------------
# 
# Training loop simply involves iterating over the training set with the
# ``DataLoader`` you created and computing the gradients, just as you
# would do for image classification or language modeling.
# 

# Create the model with given dimensions
model = GCN(dataset.dim_nfeats, 16, dataset.gclasses)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(20):
    for batched_graph, labels in train_dataloader:
        pred = model(batched_graph, batched_graph.ndata['attr'].float())
        loss = F.cross_entropy(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

num_correct = 0
num_tests = 0
for batched_graph, labels in test_dataloader:
    pred = model(batched_graph, batched_graph.ndata['attr'].float())
    num_correct += (pred.argmax(1) == labels).sum().item()
    num_tests += len(labels)

print('Test accuracy:', num_correct / num_tests)


######################################################################
# What’s next
# -----------
# 
# -  See `GIN
#    example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__
#    for an end-to-end graph classification model.
# 

