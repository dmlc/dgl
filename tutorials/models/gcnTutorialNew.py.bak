

"""
Graph Convolutional Network New
====================================
**Author**: `Qi Huang`

This is a brief entry to DGL and its message passing API through GCN(graph convolutional network).
"""

##############################################################################
# Message Passing: Warming up
# ---------------------------
#
# Let's begin with the simplest graph possible with two nodes, and set the node representations:

import torch as th
import dgl

g = dgl.DGLGraph()
g.add_nodes(2)
g.add_edge(1, 0)

x = th.tensor([[0.0, 0.0], [1.0, 2.0]])
g.set_n_repr({'x': x})

##############################################################################
# What we want to do is simply to copy representation from node#1 to node#0, but with
# a message passing interface. We do this like what we will do over a pair of sockets,
# with a ``send`` and a ``recv`` interface.
# The two `user defined function (UDF)` specifies the actions: deposit the value into an internal
# key-value store with the key `msg`, and retrive it. Note that there may be multiple incoming edges
# to a node, and the receiving end aggregates them.
#
# .. note::
#  * ``send(src, dst)`` defines an edge explictly, so ``message_func`` taking ``edge`` as an
#    argument is confusing.
#  * following graph construction semantics, it'll be nice to allow ``src`` and ``dst`` as a pair
#    of lists, or a pair of tensor, though this example doesn't demonstrate it.
#  * likewise, since we allow edge broadcasting, we should allow it in ``send`` as well.
#  * what's the side-effect of doing a send action? we are left with the impression that the second argument
#    in the ``reduce_func`` (i.e. ``msgs``) magically gets the stuff with the same key.
#  * my preference is to say that expected side-effect is simply that the result  of a ``send`` action is available
#    at ``dst['key']``, where ``key`` is whatever the user specified in ``message_func``. this allows
#    for cases where we use ``apply_node_func``.
#  * in other words,
#    ``message_func`` returns ``{'hey': [1.0]}``, we expect to see ``dst['hey']``. if that happens
#    to be the represnetation key, then a replacement is done. user can define a new key, e.g. ``accum``,
#    then the ``reduce_func`` and ``apply_node_func`` can do whatever they want. typically,
#    they should return with the representation key to perform update.
#

def send_source(src, edge):
    return {'msg': src['x']}

def simple_reduce(node, msgs):
    return {'x' : th.sum(msgs['msg'], dim=1)}

g.send(1, 0, message_func=send_source)
g.recv([0], reduce_func=simple_reduce)
print(g.get_n_repr())


##############################################################################
# Some times the computation may involve representations on the edges. Let's say we want to "amplify"
# the message:

w = th.tensor([2.0])
g.set_e_repr({'w': w})

def send_source_with_edge_weight(src, edge):
    return {'msg': src['x'] * edge['w']}

g.send(1, 0, message_func=send_source_with_edge_weight)
g.recv([0], reduce_func=simple_reduce)
print(g.get_n_repr())

##############################################################################
# Or we may need to involve the desination's representation, and here is one version:

def simple_reduce_addup(node, msgs):
    return {'x' : node['x'] + th.sum(msgs['msg'], dim=1)}

g.send(1, 0, message_func=send_source_with_edge_weight)
g.recv([0], reduce_func=simple_reduce_addup)
print(g.get_n_repr())

##############################################################################
# A slightly more complex but more flexible one is to store the reduced sum at the node under
# a different key, and then call the ``apply_node_func``:
#
# .. note::
#   that the stuff magically appear as part of node's key-value is non-intuitive.


def simple_reduce_to_accum(node, msgs):
    return {'accum' : th.sum(msgs['msg'], dim=1)}

def simple_apply(node):
    return {'x': node['x'] + node['accum']}

g.send(1, 0, message_func=send_source_with_edge_weight)
g.recv([0], reduce_func=simple_reduce_to_accum, apply_node_func=simple_apply)
print(g.get_n_repr())

##############################################################################
# The ``send`` and ``recv`` is **level-1** call in DGL, they have the finest control over routing
# the message.
#
# TODO: build a star graph (reuse the one in 2_graph.py), and use pull (or push)
#
# TODO: build a much bigger graph, explain with spMV and the use of ``update_all``
#

##############################################################################
# Model Overview
# ---------------
# Introduction
# ```````````````````
# This is a simple implementation of Kipf & Welling's Semi-Supervised Classificaton with Graph Convolutional Networks in ICLR 2017, which propose a simple yet efficient model that extends convolutional neual network from the grid structured data we all familiar and like to graphs, like social network and knowledge graph. It starts from the framework of spectral graph convolutions and makes reasonable simplifications to achieve both faster training and higher prediction accuracy. It also achieves start-of-the-art classification results on a number of graph datasets like CORA, etc. /TODO: elaborate.
# Note that this is not intended to be an end-to-end lecture on Kiph & Willing's GCN paper. In this tutorial, we aim at providing a friendly entry to showcase how to code up a contemporary NN model operating on graph structure data, and increases user's understanding of DGL's message passing API in action. For a more thorough understanding of the derivation and all details of GCN, please visit the original paper. /TODO(hq): add link.
#
# GCN in one formula
# `````````````````````
# Essentially, GCN's model boils down to the following oen formula
# :math:`H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})`
#
# The equation above describes a "graph convolution layer" in GCN.
# Essentially, :math:`H^{(l)}` denotes the lth layer in the network, :math:`\sigma` is the non-linearity, and :math:`W` is the weight matrix for this layer. :math:`D` and :math:`A`, as commonly seen, represent degree matrix and adjacency matrix, respectively. The ~ is a renormalization trick in which we add a self-connection to each node of the graph, and build the corresponding degree and adjacency matrix.
#
# The shape of the input :math:`H^{(0)}` is :math:`N \times D`, where :math:`N` is the number of nodes and :math:`D` is the number of input features. We can chain up multiple layers as such to produce a node-level representation output with shape :math:`N \times F`, where :math:`F` is the dimension of the output node feature vector.
#
# Derivation of GCN
# ``````````````````
# \TODO(hq) do we need a short description of how we departure from spectral based method and end with GCN?
# According to others, this amounts to a laplacian smoothing.
#
# Understanding GCN from Message Passing
# ````````````````````````````````````````
# Think about :math:`W^{(l)}` just as a matrix of
# filter parameters to project :math:`H^{(l)}`.
# :math:`\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}` as a symmetrical normalization of the
# adjacency matrix.
#
# Combining these two, we arrives at a must succint form of GCN :
# :math:`\sigma(\hat{A}\hat{H}^{(l)})`
# where :math:`\hat{A}` means a normalized version of
# adjacency matrix, and :math:`\hat{H}` means a
# projection of last layer's node-level representation :math:`H`.
#
# We can further formulate multiplication with the adjacency matrix as performing message passing between nodes following paths encoding in the adjacency matrix.
# To make it simple, let's denote the input signal on a graph :math:`G = (V,E)` as :math:`x \in \mathcal{R}^{|\mathcal{V}|x1}`, assume each node's feature is only a scalar.
# Then, if we calculate :math:`x_{t+1} = Ax_{t}`, it amounts to perform a message passing operation on existing edges. The ith node's new feature :math:`x_{t+1}^{i}` essentially adds up the old feature vector :math:`x_{t}`, when the corresponding node index has non-zero entry on the ith row of the adjacency matrix A, i.e. has edge connection with node i. If we multiply the resulting vector :math:`x_{t+1}` again with A, the resulting vector, :math:`A^{2}x_{t}`, will be the resulting feature vector after two rounds of message-passing is performed. In this sense, :math:`A^2` encodes 2-hop neighborhood information for each node. By k-hop neighborhood, we mean any node reachable with exactly k steps starting from the current node (if self connection is not included in the original adjacency matrix), or any node reachable within k steps from the current node if self connection is included). In another view, we can also understand :math:`A^2` as :math:`A^2_{i,j}` = OR(k){ A_{i,k} && A_{k,j}}.
#
# Nonetheless, in GCN we only use :math:`\sigma(\hat{A}\hat{H}^{(l)})` in each layer, meaning we only propagate information among each node's 1-hop neighborhood for each layer.
#
#
# Model Implementation
# ------------------------
# Warming up of message passing API
# ````````````````````````````````````
# DGL provides 3 levels of message passing API, giving user different level of control. Below we demonstrate three different levels of APIs on a simple star graph of size 10, where node 1-9 all sends information to node 0.
#
# Level 1 -- send, recv, and apply_node
# ..........................................
# The most basic level is ``send(srs,dst,message_function)``, ``recv(node,reduce_function)``, and ``apply_nodes(nodes)``.
# ``send()`` and ``recv()`` allow users to designate specific pairs of (source, destination) to pass information. ``apply_nodes()`` allow users to perform per-node computation.
#
# Three functions need to be pre-specified when using message pasing api: 1) message function 2) reduce function 3) apply function. Message function determines what message is passed along edges; reduce function determines how messages are aggregated at the destination node; apply functions determines  Note that all these three functions can be either defined by users, or use built-in functions when importing ``dgl.function``. For a more detailed description of built-in function syntax, please see \TODO(hq) add hyperref.
#
# User don't have to pass message_function and reduce_function everytime as parameters to the function if they registered them in the graph in priori, as shown in the following code.
import argparse
import time
import torch as th
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import dgl
import networkx as nx
from dgl import DGLGraph
from dgl.data import register_data_args, load_data

star = dgl.DGLGraph()
star.add_nodes(10)
u = list(range(1,10))
star.add_edges(u,0) # create the graph
D = 1  # the feature dimension
N = star.number_of_nodes()
M = star.number_of_edges()
nfeat = th.ones((N, D))  # each node's feature is just 1
efeat = th.ones((M, D))*2  # each edge's feature is 2.
star.set_n_repr({'hv' : nfeat})
star.set_e_repr({'he' : efeat})
u = th.tensor([0])
v = th.tensor([1,2,3,4,5]) #sending node 1-5's node feature to node 0's.
def _message_test(src,edge):
    return {'hv':src['hv']}
def _reduce(node,msgs):
    return{'hv':node['hv']+msgs['hv'].sum(1)}
    # aggregate alone the second dimension as
    # the first dimension is reserved for batching in DGL.
star.register_message_func(_message_test)
star.register_reduce_func(_reduce)
star.send(v,u)
# DGL supports batching send/recv and broadcasting.
star.recv(u)
#We expect to get 6 on node 0.
print(star.get_n_repr()['hv'])
##########################################################################
# Level 2 -- pull, push, and send_and_recv
# ............................................
# It could be both tedious and inefficient for user to call ``send()`` and ``recv()`` respectively. DGL comes into aid by providing a series of higher level APIs which also increase the performance by operator fusion in the backend ``/TODO(gaiyu) verify this statement please``.
# ``send_and_recv(src,dst,message_func,reduce_func,apply_func)`` is essentially a wrapper around send and receive.
# pull(node,message_func,reduce_func,apply_func) will take the input nodes as destination nodes, and all their predeseccor nodes as source nodes, and perform ``send_and_recv()``
# push(node,message_func,reduce_func,apply_func) will take the input nodes as source nodes, and all their descendant nodes as destination nodes, and perform ``send_and_recv()``
#
# Notice that apply function is usually optional in message passing APIs.
star.set_n_repr({'hv' : nfeat}) #reset node repr
star.set_e_repr({'he' : efeat}) #reset edge repr
star.send_and_recv(v,u) #note that here apply functon is left blank
print(star.get_n_repr()['hv']) # we expect to get 6 on node 0
#####################################################################
#
# Then we register the apply function.
#
def _apply_test(node):
    return {'hv':500*node['hv']}
star.register_apply_node_func(_apply_test)
star.apply_nodes(u)
print(star.get_n_repr()['hv']) #we expect to get 3000 on node 0
#########################################################################
star.set_n_repr({'hv' : nfeat}) #reset node repr
star.set_e_repr({'he' : efeat}) #reset edge repr
star.pull(u)
print(star.get_n_repr()['hv']) # we expect to get 3000 on node 0
###################################################################
star.set_n_repr({'hv' : nfeat}) #reset node repr
star.set_e_repr({'he' : efeat}) #reset edge repr
star.push(v)
print(star.get_n_repr()['hv']) # we expect to get 3000 on node 0
#######################################################################
# Level 3 -- update_all
# ..........................
# In many cases, user would like to perform message passing on all the edges simoutaneously, such as in the case of adjacency matrix multiplication in GCN. DGL also provides ``update_all()`` method to achieve this, also optimizing the performance under the hood.
star.set_n_repr({'hv' : nfeat}) #reset node repr
star.set_e_repr({'he' : efeat}) #reset edge repr
star.update_all(apply_node_func = None)
print(star.get_n_repr()['hv']) # we expect to get 10 on node 0, as we choose not to perform any apply_node functions
#
##########################################################
# Model Implementation
# ``````````````````````````````
# Model definition
# ....................
# Similar to above, we first define the message function, reduce function and apply function for GCN.
def gcn_msg(src, edge):
    return {'m' : src['h']} #return node feature

def gcn_reduce(node, msgs):
    return {'h' : th.sum(msgs['m'], 1)} # aggregate incoming node features

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation #apply a filter and non-linearity.

    def forward(self, node):
        h = self.linear(node['h'])
        if self.activation:
            h = self.activation(h)
            #raise RuntimeError(h.shape)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 mode=1):
        super(GCN, self).__init__()
        self.g = g #graph is passed as a parameter to the model
        self.dropout = dropout
        # input layer
        self.layers = nn.ModuleList([NodeApplyModule(in_feats, n_hidden, activation)])
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(NodeApplyModule(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(NodeApplyModule(n_hidden, n_classes))
        self.mode = mode # indicate DGL message passing level for subsequent use

    # Message passing in 3 levels --- level 1
    def lv1_mp(self, layer):
        nodeIdList = list(i for i in range(self.g.number_of_nodes()))
        for s in nodeIdList:
                self.g.send(s, nodeIdList, gcn_msg)
        self.g.recv(nodeIdList, gcn_reduce, layer)
        #self.g.apply_nodes(nodeIdList, layer)

    # Message passing in 3 levels --- level 2
    def lv2_mp(self, layer):
        dst = list(i for i in range(self.g.number_of_nodes()))
        self.g.pull(dst, gcn_msg, gcn_reduce, layer)

    # Message passing in 3 levels -- level 3
    def lv3_mp(self, layer):
        #nodeIdList = list(i for i in range(self.g.number_of_nodes()))
        self.g.update_all(gcn_msg, gcn_reduce, layer)
        #self.g.update_all(gcn_msg, gcn_reduce)
        #self.g.apply_nodes(nodeIdList, layer)

    # Below is the forward function

    def forward(self, features):
        self.g.set_n_repr({'h' : features})
        for layer in self.layers:
            # apply dropout
            if self.dropout:
                g.apply_nodes(apply_node_func=
                        lambda node: F.dropout(node['h'], p=self.dropout))
            assert self.mode in [1,2,3]
            if self.mode == 1 :
                self.lv1_mp(layer)
            elif self.mode == 2 :
                self.lv2_mp(layer)
            else :
                self.lv3_mp(layer)

        return self.g.pop_n_repr('h')
######################################################################
# Training & Inference
# ``````````````````````````````````
# Below we train the model and perform inference.
from dgl.data import citation_graph as citegrh
data = citegrh.load_cora()
features = th.FloatTensor(data.features)
print(type(features))
print(type(data.features))
labels = th.LongTensor(data.labels)
mask = th.ByteTensor(data.train_mask)
in_feats = features.shape[1]
n_classes = data.num_labels
n_edges = data.graph.number_of_edges()

# Some training hyperparameters for illustration
#cuda = False #Not sure whether there is cuda or not
cuda = True
th.cuda.set_device(-1)
features = features.cuda()
labels = labels.cuda()
mask = mask.cuda()

n_hidden = 16
n_layers = 1
dropout = 0
n_epochs = 200
lr = 1e-3
g = DGLGraph(data.graph)
model = GCN(g,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            F.relu,
            dropout,
            mode = 3) #level 3 message passing
model2 = GCN(g,
            in_feats,
            n_hidden,
            n_classes,
            n_layers,
            F.relu,
            dropout,
            mode = 3) #level 2 message passing
model.cuda()
model2.cuda()
# use optimizer
optimizer = th.optim.Adam(model2.parameters(), lr=lr)
# initialize graph
dur = []
for epoch in range(n_epochs):
    if epoch >=3:
        t0 = time.time()
    #forward
    logits = model2(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

        print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f} | ETputs(KTEPS) {:.2f}".format(
            epoch, loss.item(), np.mean(dur), n_edges / np.mean(dur) /1000))
