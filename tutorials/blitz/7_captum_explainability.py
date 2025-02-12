"""
Explaining Graph Neural Networks (GNNs) with Captum
===========================================

Captum is a model interpretability and understanding library for PyTorch. You can install it with

.. code:: bash

   pip install captum


By the end of this tutorial, you will be able to:

- Understand the basic concepts of GNN explainability
- Use Captum to explain the predicted node classes of a GNN
- Visualize prediction explanations by plotting node-centered subgraphs

(Time estimate: 20 minutes)

"""

######################################################################
# Overviews of GNN Explainability
# -------------------------------
# Like other deep learning models, GNNs are black box models. This might
# hinder the deployment of GNNs in risk-sensitive scenarios. Explaining
# GNNs has been an important research problem. Many GNN explainability
# methods approach this problem by identifying important nodes and edges
# in the input graphs.

# Node-centered subgraphs play a critical role in analyzing GNNs. The k-hop
# subgraph of a node fully determines the information that most k-layer GNNs
# exploit to compute the associated node representation. Many GNN explanation
# methods provide an explanation by extracting a subgraph and assigning
# importance weights to the nodes and edges of it. We will visualize
# node-centered weighted subgraphs with a built-in DGL function. This is
# beneficial for debugging and understanding GNNs and GNN explanation methods.

# For this demonstration, we will use IntegratedGradients from `Captum <https://github.com/pytorch/captum>`__ to
# explain the predictions of a graph convolutional network (GCN).
# Specifically, we try to find the most important nodes and edges to the
# model classifying nodes in a graph.
#


######################################################################
# Loading Cora Dataset
# --------------------
#
# First, we load DGL’s built-in Cora dataset and retrieve its graph
# structure, node labels (classes) and the number of node classes.
# 

# Install and import required packages.
import dgl
# The Cora dataset used in this tutorial only consists of one single graph.
dataset = dgl.data.CoraGraphDataset()
g = dataset[0]
g = dgl.add_self_loop(g)


######################################################################
# Define the model
# ----------------
# Then, we will build a two-layer Graph Convolutional Network (GCN).
# Each layer computes new node representations by aggregating neighbor information.
# What's more, we use GraphConv which supports ``edge_weight`` as a
# parameter to calculate the edge importance in model explanation.
# 

import torch.nn as nn
from dgl.nn import GraphConv
# Define a class for GCN
class GCN(nn.Module):
    def __init__(self, in_feats, num_classes, h_feats=16):
        super().__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats, num_classes)

    # The parameter `edge_weight` will weight the messages in message passing for edges
    def forward(self, in_feat, g, edge_weight=None, nid=None):
        h = self.conv1(g, in_feat, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight=edge_weight)
        # nid is used to identify the target node
        if nid is None:
            return h
        else:
            return h[nid:nid + 1]


######################################################################
# Training the model
# ------------------
#

import torch
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = g.to(device)
features = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

model = GCN(features.shape[1], dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    # Forward
    logits = model(features, g)
    
    # Compute loss
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    
    # Backward
    loss.backward()
    optimizer.step()


######################################################################
# Explaining the predictions
# --------------------------
# We will explain why the model might make the prediction of class C for all nodes.
# C is the ground truth label of node 10 for demonstration purpose.
#
# First, we assign importance to the nodes by attributing the prediction of class C
# for all nodes to the input node features with IntegratedGradients.
#

# Select the node with index 10 for demonstration
output_idx = 10
target = int(labels[output_idx])
print(target)


######################################################################
# Since the ``IntergratedGradients`` method only allows one argument
# to be passed, we use ``partial`` function to pass the default value
# to the forward function.
#

from captum.attr import IntegratedGradients
from functools import partial

ig = IntegratedGradients(partial(model.forward, g=g, nid=output_idx))
# Attribute the predictions of node class C to the input node features
ig_attr_node = ig.attribute(g.ndata['feat'], target=target,
                            internal_batch_size=g.num_nodes(), n_steps=50)
print(ig_attr_node.shape)


######################################################################
# We compute the node importance weights from the input feature weights
# and normalize them.
#

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()


######################################################################
# We visualize a node-centered subgraph with node weights
#

# Visualize
from utility import visualize_subgraph
import matplotlib.pyplot as plt

num_hops = 2
ax, nx_g = visualize_subgraph(g, output_idx, num_hops, node_alpha=ig_attr_node)
plt.show()


######################################################################
# Then, we will similarly assign importance weights to edges.
#

def model_forward(edge_mask, g, nid):
    return model(g.ndata['feat'], g, edge_weight=edge_mask, nid=nid)

# Initialize an edge mask so that we can attribute to the edges, which is similar to an adjacency matrix
edge_mask = torch.ones(g.num_edges()).to(device)
ig = IntegratedGradients(partial(model_forward, g=g, nid=output_idx))
ig_attr_edge = ig.attribute(edge_mask, target=target, internal_batch_size=g.num_edges(), n_steps=50)
print(ig_attr_edge.shape)

# Scale attributions to [0, 1]:
ig_attr_edge = ig_attr_edge.abs()
ig_attr_edge /= ig_attr_edge.max()


######################################################################
# We visualize a node-centered subgraph with edge weights
#

ax, nx_g = visualize_subgraph(g, output_idx, num_hops, edge_alpha=ig_attr_edge)
plt.show()


######################################################################
# Visualize both node and edge weights
# 

ax, nx_g = visualize_subgraph(g, output_idx, num_hops, node_alpha=ig_attr_node, edge_alpha=ig_attr_edge)
plt.show()


######################################################################
# We can also restruct a new forward function to calculate both node
# and edge importance weights at the same time.
#

def combine_model_forward(in_feat, edge_mask, g, nid):
    return model(in_feat, g, edge_weight=edge_mask, nid=nid)

edge_mask = torch.ones(g.num_edges()).to(device)
ig = IntegratedGradients(combine_model_forward)
ig_attr_node, ig_attr_edge = ig.attribute((features, edge_mask), target=target, additional_forward_args=(g, output_idx),
                                          internal_batch_size=1)

# Scale attributions to [0, 1]:
ig_attr_node = ig_attr_node.abs().sum(dim=1)
ig_attr_node /= ig_attr_node.max()
ig_attr_edge = ig_attr_edge.abs()
ig_attr_edge /= ig_attr_edge.max()

ax, nx_g = visualize_subgraph(g, output_idx, num_hops, node_alpha=ig_attr_node, edge_alpha=ig_attr_edge)
plt.show()