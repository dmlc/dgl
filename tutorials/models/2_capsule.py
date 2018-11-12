"""
.. _model-capsule:

Capsule Network Tutorial
===========================

**Author**: `Jinjing Zhou`, `Zheng Zhang`

It is perhaps a little surprising that some of the more classical models can also be described in terms of graphs,
offering a different perspective.
This tutorial describes how this is done for the `capsule network <http://arxiv.org/abs/1710.09829>`__.
"""
#######################################################################################
# Key ideas of Capsule
# --------------------
#
# There are two key ideas that the Capsule model offers.
#
# **Richer representations** In classic convolutional network, a scalar
# value represents the activation of a given feature. Instead, a capsule
# outputs a vector, whose norm represents the probability of a feature,
# and the orientation its properties.
#
# .. figure:: https://i.imgur.com/55Ovkdh.png
#    :alt:
#
# **Dynamic routing** To generalize max-pooling, there is another
# interesting proposed by the authors, as a representational more powerful
# way to construct higher level feature from its low levels. Consider a
# capsule :math:`u_i`. The way :math:`u_i` is integrated to the next level
# capsules take two steps:
#
# 1. :math:`u_i` projects differently to different higher level capsules
#    via a linear transformation: :math:`\hat{u}_{j|i} = W_{ij}u_i`.
# 2. :math:`\hat{u}_{j|i}` routes to the higher level capsules by
#    spreading itself with a weighted sum, and the weight is dynamically
#    determined by iteratively modify the and checking against the
#    "consistency" between :math:`\hat{u}_{j|i}` and :math:`v_j`, for any
#    :math:`v_j`. Note that this is similar to a k-means algorithm or
#    `competive
#    learning <https://en.wikipedia.org/wiki/Competitive_learning>`__ in
#    spirit. At the end of iterations, :math:`v_j` now integrates the
#    lower level capsules.
#
# The full algorithm is the following: |image0|
#
# The dynamic routing step can be naturally expressed as a graph
# algorithm. This is the focus of this tutorial. Our implementation is
# adapted from `Cedric
# Chee <https://github.com/cedrickchee/capsule-net-pytorch>`__, replacing
# only the routing layer, and achieving similar speed and accuracy.
#
# Model Implementation
# -----------------------------------
# Step 1: Setup and Graph Initialiation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The below figure shows the directed bipartitie graph built for capsules
# network. We denote :math:`b_{ij}`, :math:`\hat{u}_{j|i}` as edge
# features and :math:`v_j` as node features. |image1|
#
import torch.nn as nn
import torch as th
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import dgl


def init_graph(in_nodes, out_nodes, f_size, u_hat):
    g = dgl.DGLGraph()
    all_nodes = in_nodes + out_nodes
    g.add_nodes(all_nodes)

    in_indx = list(range(in_nodes))
    out_indx = list(range(in_nodes, in_nodes + out_nodes))
    # add edges use edge broadcasting
    for u in in_indx:
        g.add_edges(u, out_indx)

    # init states
    g.ndata['v'] = th.zeros(all_nodes, f_size)
    g.edata['u_hat'] = u_hat
    g.edata['b'] = th.zeros(in_nodes * out_nodes, 1)
    return g


#########################################################################################
# Step 2: Define message passing functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Recall the following steps, and they are implemented in the class
# ``DGLRoutingLayer`` as the followings:
#
# 1. Normalize over out edges
#
#    -  Softmax over all out-edge of in-capsules
#       :math:`\textbf{c}_i = \text{softmax}(\textbf{b}_i)`.
#
# 2. Weighted sum over all in-capsules
#
#    -  Out-capsules equals weighted sum of in-capsules
#       :math:`s_j=\sum_i c_{ij}\hat{u}_{j|i}`
#
# 3. Squash Operation
#
#    -  Squashing function is to ensure that short capsule vectors get
#       shrunk to almost zero length while the long capsule vectors get
#       shrunk to a length slightly below 1. Its norm is expected to
#       represents probabilities at some levels.
#    -  :math:`v_j=\text{squash}(s_j)=\frac{||s_j||^2}{1+||s_j||^2}\frac{s_j}{||s_j||}`
#
# 4. Update weights by agreement
#
#    -  :math:`\hat{u}_{j|i}\cdot v_j` can be considered as agreement
#       between current capsule and updated capsule,
#       :math:`b_{ij}=b_{ij}+\hat{u}_{j|i}\cdot v_j`
class DGLRoutingLayer(nn.Module):
    def __init__(self, in_nodes, out_nodes, f_size, u_hat):
        super(DGLRoutingLayer, self).__init__()
        self.g = init_graph(in_nodes, out_nodes, f_size, u_hat)
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.in_indx = list(range(in_nodes))
        self.out_indx = list(range(in_nodes, in_nodes + out_nodes))

    def forward(self, routing_num=1):
        for r in range(routing_num):
            # step 1 (line 4): normalize over out edges
            in_edges = self.g.edata['b'].view(self.in_nodes, self.out_nodes)
            self.g.edata['c'] = F.softmax(in_edges, dim=1).view(-1, 1)

            def cap_message(edges):
                return {'m': edges.data['c'] * edges.data['u_hat']}
            self.g.register_message_func(cap_message)

            # step 2 (line 5)
            def cap_reduce(nodes):
                return {'s': th.sum(nodes.mailbox['m'], dim=1)}
            self.g.register_reduce_func(cap_reduce)

            # Execute step 1 & 2
            self.g.update_all()

            # step 3 (line 6)
            self.g.nodes[self.out_indx].data['v'] = self.squash(self.g.nodes[self.out_indx].data['s'], dim=1)

            # step 4 (line 7)
            v = th.cat([self.g.nodes[self.out_indx].data['v']] * self.in_nodes, dim=0)
            self.g.edata['b'] = self.g.edata['b'] + (self.g.edata['u_hat'] * v).sum(dim=1, keepdim=True)

    @staticmethod
    def squash(s, dim=1):
        sq = th.sum(s ** 2, dim=dim, keepdim=True)
        s_norm = th.sqrt(sq)
        s = (sq / (1.0 + sq)) * (s / s_norm)
        return s


############################################################################################################
# Step 3: Testing
# ~~~~~~~~~~~~~~~
#
# Let's make a simple 20x10 capsule layer:
in_nodes = 20
out_nodes = 10
f_size = 4
u_hat = th.randn(in_nodes * out_nodes, f_size)
routing = DGLRoutingLayer(in_nodes, out_nodes, f_size, u_hat)

############################################################################################################
# We can visualize the behavior by monitoring the entropy of outgoing
# weights, they should start high and then drop, as the assignment
# gradually concentrate:
entropy_list = []
dist_list = []

for i in range(10):
    routing(1)
    dist_matrix = routing.g.edata['c'].view(in_nodes, out_nodes)
    entropy = (-dist_matrix * th.log(dist_matrix)).sum(dim=1)
    entropy_list.append(entropy.data.numpy())
    dist_list.append(dist_matrix.data.numpy())

############################################################################################################
#
# .. figure:: https://i.imgur.com/dMvu7p3.png
#    :alt:
#
# Alternatively, we can also watch the evolution of histograms: |image2|
#
# Or monitor the how lower level capcules gradually attach to one of the
# higher level ones: |image3|
#
# The full code of this visulization is provided at (link); the complete
# code that trains on MNIST is at (link).
#
# .. |image0| image:: https://i.imgur.com/mv1W9Rv.png
# .. |image1| image:: https://i.imgur.com/9tc6GLl.png
# .. |image2| image:: https://github.com/VoVAllen/DGL_Capsule/raw/master/routing_dist.gif
# .. |image3| image:: https://github.com/VoVAllen/DGL_Capsule/raw/master/routing_vis.gif
#
