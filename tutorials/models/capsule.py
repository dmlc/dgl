"""
Capsule Network
================

**Author**: `Jinjing Zhou`
 
This tutorial explains how to use DGL library and its language to implement the
`capsule network <http://arxiv.org/abs/1710.09829>`__ proposed by Geoffrey Hinton and his team.
The algorithm aims to provide a better alternative to current neural network structures.
By using DGL library, users can implement the algorithm in a more intuitive way.
"""


##############################################################################
# Model Overview
# ---------------
# Introduction
# ```````````````````
# Capsule Network were first introduced in 2011 by Geoffrey Hinton, et al.,
# in paper `Transforming Autoencoders <https://www.cs.toronto.edu/~fritz/absps/transauto6.pdf>`__,
# but it was only a few months ago, in November 2017, that Sara Sabour, Nicholas Frosst,
# and Geoffrey Hinton published a paper called Dynamic Routing between Capsules, where they
# introduced a CapsNet architecture that reached state-of-the-art performance on MNIST.
#  
# What's a capsule?
# ```````````````````
# In papers, author states that "A capsule is a group of neurons whose activity vector
# represents the instantiation parameters of a specific type of entity such as an object
# or an object part."
#
# Generally speaking, the idea of capsule is to encode all the information about the
# features into a vector form, by substituting scalars in traditional neural network with vectors.
# And use the norm of the vector to represents the meaning of original scalars. 
# 
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/capsule_f1.png
# 
# Dynamic Routing Algorithm
# `````````````````````````````
# Due to the different structure of network, capsules network has different operations to
# calculate results. This figure shows the comparison, drawn by
# `Max Pechyonkin <https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-ii-how-capsules-work-153b6ade9f66O>`__
# 
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/capsule_f2.png
#    :height: 250px
# 
# The key idea is that the output of each capsule is the sum of weighted input vectors.
# We will go into details in the later section with code implementations.
# 
# Model Implementations
# -------------------------

##############################################################################
# Algorithm Overview
# ```````````````````````````
#
# .. image:: https://raw.githubusercontent.com/VoVAllen/DGL_Capsule/master/algorithm.png
#
# The main step of routing algorithm is line 4 - 7. In ``DGLGraph`` structure, we consider these steps as a message passing
# procedure.

##############################################################################
# Consider capsule routing as a graph structure
# ````````````````````````````````````````````````````````````````````````````
# We can consider each capsule as a node in a graph, and connect all the nodes between layers.
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/capsule_f3.png
#    :height: 150px
#
def construct_graph(self):
    g = dgl.DGLGraph()
    g.add_nodes(self.input_capsule_num + self.output_capsule_num)
    input_nodes = list(range(self.input_capsule_num))
    output_nodes = list(range(self.input_capsule_num, self.input_capsule_num + self.output_capsule_num))
    u, v = [], []
    for i in input_nodes:
        for j in output_nodes:
            u.append(i)
            v.append(j)
    g.add_edges(u, v)
    return g, input_nodes, output_nodes


##############################################################################
# Write Message Passing Functions
# ``````````````````````````````````
# Reduce Functions (line 4 - 5)
# .............................................
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/capsule_f5.png
#
# At this stage, we need to define a reduce function to aggregate all the information we
# get from layer :math:`l` and set into layer :math:`(l+1)`'s node features.
#
# .. note::
#    The softmax operation is over dimension :math:`j` instead of :math:`i`.
def capsule_reduce(node, msg):
    b_ij_c, u_hat = msg['b_ij'], msg['u_hat']
    # line 4
    c_i = F.softmax(b_ij_c, dim=0)
    # line 5
    s_j = (c_i.unsqueeze(2).unsqueeze(3) * u_hat).sum(dim=1)
    return {'h': s_j}


##############################################################################
# Node Update Functions (line 6)
# ......................................................
# Squash the intermediate representations into node features :math:`v_j`
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/step6.png
#
def capsule_update(msg):
    v_j = squash(msg['h'])
    return {'h': v_j}


##############################################################################
# Edge Update Functions (line 7)
# ...........................................................................
# Update the routing parameters by updating edges in graph
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/step7.png
#
def update_edge(u, v, edge):
    return {'b_ij': edge['b_ij'] + (v['h'] * edge['u_hat']).mean(dim=1).sum(dim=1)}


##############################################################################
# Call DGL function to execute algorithm
# ````````````````````````````````````````````````````````````````````````````
# Call ``update_all`` and ``update_edge`` functions to execute the whole algorithms.
# Message function is to define which attributes are needed in further computations
#
def routing(self):
    def capsule_msg(src, edge):
        return {'b_ij': edge['b_ij'], 'h': src['h'], 'u_hat': edge['u_hat']}

    self.g.update_all(capsule_msg, capsule_reduce, capsule_update)
    self.g.update_edge(edge_func=update_edge)


##############################################################################
# Forward Function
# ````````````````````````````````````````````````````````````````````````````
# This section shows the whole process of forward process of capsule routing algorithm.
def forward(self, x):
    self.batch_size = x.size(0)
    u_hat = self.compute_uhat(x)
    self.initialize_nodes_and_edges_features(u_hat)
    for i in range(self.num_routing):
        self.routing()
    this_layer_nodes_feature = self.g.get_n_repr()['h'][
                               self.input_capsule_num:self.input_capsule_num + self.output_capsule_num]
    return this_layer_nodes_feature.transpose(0, 1).unsqueeze(1).unsqueeze(4).squeeze(1)


##############################################################################
# Initialization & Affine Transformation
# ````````````````````````````````````````````````````````````````````````````
# - Pre-compute :math:`\hat{u}_{j|i}`, initialize :math:`b_{ij}` and store them as edge attribute
# - Initialize node features as zero
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/capsule_f4.png
#

def compute_uhat(self, x):
    # x is the input vextor with shape [batch_size, input_capsule_dim, input_num]
    # Transpose x to [batch_size, input_num, input_capsule_dim]
    x = x.transpose(1, 2)
    # Expand x to [batch_size, input_num, output_num, input_capsule_dim, 1]
    x = torch.stack([x] * self.output_capsule_num, dim=2).unsqueeze(4)
    # Expand W from [input_num, output_num, input_capsule_dim, output_capsule_dim]
    # to [batch_size, input_num, output_num, output_capsule_dim, input_capsule_dim]
    W = self.weight.expand(self.batch_size, *self.weight.size())
    # u_hat's shape is [input_num, output_num, batch_size, output_capsule_dim]
    u_hat = torch.matmul(W, x).permute(1, 2, 0, 3, 4).squeeze().contiguous()
    return u_hat


def initialize_nodes_and_edges_features(self, u_hat):
    b_ij = torch.zeros(self.input_capsule_num, self.output_capsule_num).to(self.device)
    self.g.set_e_repr({'b_ij': b_ij.view(-1)})
    self.g.set_e_repr({'u_hat': u_hat.view(-1, self.batch_size, self.output_capsule_dim)})

    # Initialize all node features as zero
    node_features = torch.zeros(self.input_capsule_num + self.output_capsule_num, self.batch_size,
                                self.output_capsule_dim).to(self.device)
    self.g.set_n_repr({'h': node_features})


##############################################################################
# Squash function
# ..................
# Squashing function is to ensure that short vectors get shrunk to almost zero length and
# long vectors get shrunk to a length slightly below 1. Its norm is expected to represents probabilities
# at some levels.
#
# .. image:: https://raw.githubusercontent.com/dmlc/web-data/master/dgl/tutorials/capsule/squash.png
#    :height: 100px
#
def squash(s, dim=2):
    sq = torch.sum(s ** 2, dim=dim, keepdim=True)
    s_std = torch.sqrt(sq)
    s = (sq / (1.0 + sq)) * (s / s_std)
    return s


##############################################################################
# General Setup
# `````````````````````````

import dgl
import torch
import torch.nn.functional as F
from torch import nn


class DGLDigitCapsuleLayer(nn.Module):
    def __init__(self, input_capsule_dim=8, input_capsule_num=1152, output_capsule_num=10, output_capsule_dim=16,
                 num_routing=3, device='cpu'):
        super(DGLDigitCapsuleLayer, self).__init__()
        self.device = device
        self.input_capsule_dim = input_capsule_dim
        self.input_capsule_num = input_capsule_num
        self.output_capsule_dim = output_capsule_dim
        self.output_capsule_num = output_capsule_num
        self.num_routing = num_routing
        self.weight = nn.Parameter(
            torch.randn(input_capsule_num, output_capsule_num, output_capsule_dim, input_capsule_dim))
        self.g, self.input_nodes, self.output_nodes = self.construct_graph()


# This section is for defining class in multiple cells.
DGLDigitCapsuleLayer.construct_graph = construct_graph
DGLDigitCapsuleLayer.forward = forward
DGLDigitCapsuleLayer.routing = routing
DGLDigitCapsuleLayer.compute_uhat = compute_uhat
DGLDigitCapsuleLayer.initialize_nodes_and_edges_features = initialize_nodes_and_edges_features
