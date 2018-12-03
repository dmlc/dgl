"""
.. currentmodule:: dgl

DGL at a Glance
=========================

**Author**: `Minjie Wang <https://jermainewang.github.io/>`_, Quan Gan, `Jake
Zhao <https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang

The goal of this tutorial:

- Understand how DGL builds a graph and performs computation on graph from a
  high level.
- Train a simple graph neural network in DGL to classify nodes in a graph.

At the end of this tutorial, we hope you get a brief feeling of how DGL works.
"""

###############################################################################
# Why DGL?
# ----------------
# DGL is designed to bring **machine learning** closer to **graph-structured
# data**. Specifically DGL enables trouble-free implementation of graph neural
# network (GNN) model family. Unlike PyTorch or Tensorflow, DGL provides
# friendly APIs to perform the fundamental operations in GNNs such as message
# passing and reduction. Through DGL, we hope to benefit both researchers
# trying out new ideas and engineers in production. 
#
# *This tutorial assumes basic familiarity with pytorch.*

###############################################################################
# A toy graph: Zachary's Karate Club
# ----------------------------------
#
# We start by creating the well-knowned "Zachary's karate club" social network.
# The network captures 34 members of a karate club, documenting pairwise links
# between members who interacted outside the club. The club later splits into
# two communities led by the instructor (node 0) and the club president (node
# 33). A visualization of the network and the community is as follows:
#
# .. image:: http://historicaldataninjas.com/wp-content/uploads/2014/05/karate.jpg 
#    :height: 400px
#    :width: 500px
#    :align: center
#
# Out task is to **build a graph neural network to predict which side each
# member will join.**


###############################################################################
# Build the graph
# ---------------
# A graph is built using :class:`~dgl.DGLGraph` class. Here is how we add the 34 members
# and their interaction edges into the graph.

import dgl

def build_karate_club_graph():
    g = dgl.DGLGraph()
    # add 34 nodes into the graph; nodes are labeled from 0~33
    g.add_nodes(34)
    # all the 78 edges in a list of tuple
    edge_list = [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2),
        (4, 0), (5, 0), (6, 0), (6, 4), (6, 5), (7, 0), (7, 1),
        (7, 2), (7, 3), (8, 0), (8, 2), (9, 2), (10, 0), (10, 4),
        (10, 5), (11, 0), (12, 0), (12, 3), (13, 0), (13, 1), (13, 2),
        (13, 3), (16, 5), (16, 6), (17, 0), (17, 1), (19, 0), (19, 1),
        (21, 0), (21, 1), (25, 23), (25, 24), (27, 2), (27, 23),
        (27, 24), (28, 2), (29, 23), (29, 26), (30, 1), (30, 8),
        (31, 0), (31, 24), (31, 25), (31, 28), (32, 2), (32, 8),
        (32, 14), (32, 15), (32, 18), (32, 20), (32, 22), (32, 23),
        (32, 29), (32, 30), (32, 31), (33, 8), (33, 9), (33, 13),
        (33, 14), (33, 15), (33, 18), (33, 19), (33, 20), (33, 22),
        (33, 23), (33, 26), (33, 27), (33, 28), (33, 29), (33, 30),
        (33, 31), (33, 32)]
    # edges in DGL is added by two list of nodes: src and dst
    src, dst = tuple(zip(*edge_list))
    g.add_edges(src, dst)
    # edges are directional in DGL; make it bi-directional
    g.add_edges(dst, src)

    return g

###############################################################################
# We can test it to see we have the correct number of nodes and edges:

G = build_karate_club_graph()
print('We have %d nodes.' % G.number_of_nodes())
print('We have %d edges.' % G.number_of_edges())

###############################################################################
# We can also visualize it by converting it to a `networkx
# <https://networkx.github.io/documentation/stable/>`_ graph:

import networkx as nx
nx_G = G.to_networkx()
pos = nx.circular_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True)

###############################################################################
# Assign features
# ---------------
# Features are tensor data associated with nodes and edges. The features of
# mulitple nodes/edges are batched along the first dimension. Following codes
# assign a one-hot encoding feature for each node in the graph (e.g. :math:`v_i` got
# a feature vector :math:`[0,\ldots,1,\dots,0]`, where the :math:`i^{th}` location is one).

import torch

G.ndata['feat'] = torch.eye(34)


###############################################################################
# We can print out the node features to verify:

# print out node 2's input feature
print(G.nodes[2].data['feat'])

# print out node 10 and 11's input features
print(G.nodes[[10, 11]].data['feat'])

###############################################################################
# Define a Graph Convolutional Network (GCN)
# ------------------------------------------
# To classify whose side each node will join, we adopt the Graph Convolutional
# Network (GCN) developed by `Kipf and
# Welling <https://arxiv.org/abs/1609.02907>`_. The GCN model can be summarized,
# in a high-level as follows:
#
# - Each node :math:`v_i` has a feature vector :math:`h_i`.
# - Each node accumulates the feature vectors :math:`h_j` from its neighbors, performs
#   an affine and non-linear transformation to update its own feature.
#
# A graphical demonstration is displayed below.
#
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/mailbox.png
#    :alt: mailbox
#    :align: center
#
# The GCN layer can be easily implemented in DGL using the message passing
# interface. It typically consists of three steps:
#
# 1. Define the message function.
# 2. Define the reduce function.
# 3. Define how they are triggered using message passing APIs (e.g. ``send`` and ``recv``).
#
# Following is how it looks like:

import torch.nn as nn
import torch.nn.functional as F

# Define the message & reduce function
# NOTE: we ignore the normalization constant c_ij for this tutorial.
def gcn_message(edges):
    # The argument is a batch of edges.
    # This computes a message called 'msg' using the source node's feature 'h'.
    return {'msg' : edges.src['h']}

def gcn_reduce(nodes):
    # The argument is a batch of nodes.
    # This computes the new 'h' features by summing the received 'msg'
    # in mailbox.
    return {'h' : torch.sum(nodes.mailbox['msg'], dim=1)}

# Define the GCNLayer module
class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, inputs):
        # g is the graph and the inputs is the input node features
        # first set the node features
        g.ndata['h'] = inputs
        # trigger message passing on all the edges and nodes
        g.send(g.edges(), gcn_message)
        g.recv(g.nodes(), gcn_reduce)
        # get the result node features
        h = g.ndata.pop('h')
        # perform linear transformation
        return self.linear(h)

###############################################################################
# We then define a neural network that contains two GCN layers:

# Define a 2-layer GCN model
class Net(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(Net, self).__init__()
        self.gcn1 = GCNLayer(in_feats, hidden_size)
        self.gcn2 = GCNLayer(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.gcn1(g, inputs)
        h = torch.relu(h)
        h = self.gcn2(g, h)
        return h
# input_feature_size=34, hidden_size=5, num_classes=2
net = Net(34, 5, 2)

###############################################################################
# Train the GCN model to predict community
# ----------------------------------------
#
# To prepare the input features and labels, again, we adopt a 
# semi-supervised setting. Each node is initialized by an
# one-hot encoding, and only the instructor (node 0) and the club president
# (node 33) are labeled.

inputs = torch.eye(34)
labeled_nodes = torch.tensor([0, 33])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

###############################################################################
# The training loop is no fancier than other NN models. We (1) create an optimizer,
# (2) feed the inputs to the model, (3) calculate the loss and (4) use autograd
# to optimize the model.

optimizer = torch.optim.Adam(net.parameters(), lr=0.01)
all_logits = []
for epoch in range(30):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))

###############################################################################
# Since the model produces a 2-dimensional vector for each node, we can
# visualize it very easily.
import matplotlib.animation as animation
import matplotlib.pyplot as plt

def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(34):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

###############################################################################
# We first plot the initial guess before training. As you can see, the nodes
# are not classified correctly.

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(0)  # draw the prediction of the first epoch
plt.close()

###############################################################################
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/karate0.png
#    :height: 300px
#    :width: 400px
#    :align: center

###############################################################################
# The following animation shows how the model correctly predicts the community
# after training.

ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

###############################################################################
# .. image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/1_first/karate.gif
#    :height: 300px
#    :width: 400px
#    :align: center

###############################################################################
# Next steps
# ----------
# In the :doc:`next tutorial <2_basics>`, we will go through some more basics
# of DGL, such as reading and writing node/edge features.
