"""
Link Prediction using Graph Neural Networks
===========================================

In the :doc:`introduction <1_introduction>`, you have already learned the
basic workflow of using GNNs for node classification, i.e. predicting
the category of a node in a graph. This tutorial will teach you how to
train a GNN for link prediction, i.e. predicting the existence of an
edge between two arbitrary nodes in a graph.

By the end of this tutorial you will be able to

-  Build a GNN-based link prediction model.
-  Train and evaluate the model on a small DGL-provided dataset.

"""

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import scipy.sparse as sp


######################################################################
# Overview of Link Prediction with GNN
# ------------------------------------
# 
# Many applications such as social recommendation, item recommendation,
# knowledge graph completion, etc., can be formulated as link prediction,
# which predicts whether an edge exists between two particular nodes. This
# tutorial shows an example of predicting whether a citation relationship,
# either citing or being cited, between two papers exists in a citation
# network.
# 
# This tutorial follows a relatively simple practice from
# `SEAL <https://papers.nips.cc/paper/2018/file/53f0d7c537d99b3824f0f99d62ea2428-Paper.pdf>`__.
# It formulates the link prediction problem as a binary classification
# problem as follows:
# 
# -  Treat the edges in the graph as *positive examples*.
# -  Sample a number of non-existent edges (i.e. node pairs with no edges
#    between them) as *negative* examples.
# -  Divide the positive examples and negative examples into a training
#    set and a test set.
# -  Evaluate the model with any binary classification metric such as Area
#    Under Curve (AUC).
# 
# In some domains such as large-scale recommender systems or information
# retrieval, you may favor metrics that emphasize good performance of
# top-K predictions. In these cases you may want to consider other metrics
# such as mean average precision, and use other negative sampling methods,
# which are beyond the scope of this tutorial.
# 
# Loading graph and features
# --------------------------
# 
# Following the :doc:`introduction <1_introduction>`, we first load the
# Cora dataset.
# 

import dgl.data

dataset = dgl.data.CoraGraphDataset()
g = dataset[0]


######################################################################
# Preparing training and testing sets
# -----------------------------------
# 
# This tutorial randomly picks 10% of the edges for positive examples in
# the test set, and leave the rest for the training set. It then samples
# the same number of edges for negative examples in both sets.
# 

# Split edge set for training and testing
u, v = g.edges()

eids = np.arange(g.number_of_edges())
eids = np.random.permutation(eids)
test_size = int(len(eids) * 0.1)
train_size = g.number_of_edges() - test_size
test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

# Find all negative edges and split them for training and testing
adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
adj_neg = 1 - adj.todense() - np.eye(g.number_of_nodes())
neg_u, neg_v = np.where(adj_neg != 0)

neg_eids = np.random.choice(len(neg_u), g.number_of_edges() // 2)
test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]

# Create training set.
train_u = torch.cat([torch.as_tensor(train_pos_u), torch.as_tensor(train_neg_u)])
train_v = torch.cat([torch.as_tensor(train_pos_v), torch.as_tensor(train_neg_v)])
train_label = torch.cat([torch.zeros(len(train_pos_u)), torch.ones(len(train_neg_u))])

# Create testing set.
test_u = torch.cat([torch.as_tensor(test_pos_u), torch.as_tensor(test_neg_u)])
test_v = torch.cat([torch.as_tensor(test_pos_v), torch.as_tensor(test_neg_v)])
test_label = torch.cat([torch.zeros(len(test_pos_u)), torch.ones(len(test_neg_u))])


######################################################################
# When training, you will need to remove the edges in the test set from
# the original graph. You can do this via ``dgl.remove_edges``.
#
# .. note::
#
#    ``dgl.remove_edges`` works by creating a subgraph from the original
#    graph, resulting in a copy and therefore could be slow for large
#    graphs.  If so, you could save the training and test graph to
#    disk, as you would do for preprocessing.
# 

train_g = dgl.remove_edges(g, eids[:test_size])


######################################################################
# Defining a GraphSAGE model
# --------------------------
# 
# This tutorial builds a model consisting of two
# `GraphSAGE <https://arxiv.org/abs/1706.02216>`__ layers, each computes
# new node representations by averaging neighbor information. DGL provides
# ``dgl.nn.SAGEConv`` that conveniently creates a GraphSAGE layer.
# 

from dgl.nn import SAGEConv

# ----------- 2. create model -------------- #
# build a two-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
    
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h
    
model = GraphSAGE(train_g.ndata['feat'].shape[1], 16)


######################################################################
# The model then predicts the probability of existence of an edge by
# computing a dot product between the representations of both incident
# nodes.
# 
# .. math::
# 
# 
#    \hat{y}_{u\sim v} = \sigma(h_u^T h_v)
# 
# The loss function is simply binary cross entropy loss.
# 
# .. math::
# 
# 
#    \mathcal{L} = -\sum_{u\sim v\in \mathcal{D}}\left( y_{u\sim v}\log(\hat{y}_{u\sim v}) + (1-y_{u\sim v})\log(1-\hat{y}_{u\sim v})) \right)
# 
# .. note::
# 
#    This tutorial does not include evaluation on a validation
#    set. In practice you should save and evaluate the best model based on
#    performance on the validation set.
# 

# ----------- 3. set up loss and optimizer -------------- #
# in this case, loss will in training loop
optimizer = torch.optim.Adam(itertools.chain(model.parameters()), lr=0.01)

# ----------- 4. training -------------------------------- #
all_logits = []
for e in range(100):
    # forward
    logits = model(train_g, train_g.ndata['feat'])
    pred = torch.sigmoid((logits[train_u] * logits[train_v]).sum(dim=1))
    
    # compute loss
    loss = F.binary_cross_entropy(pred, train_label)
    
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    all_logits.append(logits.detach())
    
    if e % 5 == 0:
        print('In epoch {}, loss: {}'.format(e, loss))

# ----------- 5. check results ------------------------ #
from sklearn.metrics import roc_auc_score
with torch.no_grad():
    pred = torch.sigmoid((logits[test_u] * logits[test_v]).sum(dim=1))
    pred = pred.numpy()
    label = test_label.numpy()
    print('AUC', roc_auc_score(label, pred))


