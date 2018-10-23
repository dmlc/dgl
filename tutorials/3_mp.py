"""
.. _tutorial-mp:

Message passing on graph
========================
**Author**: `Minjie Wang <https://jermainewang.github.io/>`_

Many of the graph-based deep neural networks are based on *"message passing"* --
nodes compute messages that are sent to others and the features are updated
using the messages. In this tutorial, we introduce the basic mechanism of message
passing in DGL.
"""

###############################################################################
# Let us start by import DGL and create an example graph used throughput this
# tutorial. The graph has 10 nodes, with node#0 be the source and node#9 be the
# sink. The source node (node#0) connects to all other nodes besides the sink
# node. Similarly, the sink node is connected by all other nodes besides the
# source node. We also initialize the feature vector of the source node to be
# all one, while the others have features of all zero.
# The code to create such graph is as follows (using pytorch syntax):

import dgl
import torch as th

g = dgl.DGLGraph()
g.add_nodes(10)
g.add_edges(0, list(range(1, 9)))
g.add_edges(list(range(1, 9)), 9)
# TODO(minjie): plot the graph here.
N = g.number_of_nodes()
M = g.number_of_edges()
print('#Nodes:', N)
print('#Edges:', M)
# initialize the node features
D = 1  # feature size
g.set_n_repr({'feat' : th.zeros((N, D))})
g.set_n_repr({'feat' : th.ones((1, D))}, 0)
print(g.get_n_repr()['feat'])

###############################################################################
# User-defined functions and high-level APIs
# ------------------------------------------
#
# There are two core components in DGL's message passing programming model:
#
# * **User-defined functions (UDFs)** on how the messages are computed and used.
# * **High-level APIs** on who are sending messages to whom and are being updated.
#
# For example, one simple user-defined message function can be as follows:

def send_source(src, edge):
    return {'msg' : src['feat']}

###############################################################################
# The above function computes the messages over **a batch of edges**.
# It has two arguments: `src` for source node features and
# `edge` for the edge features, and it returns the messages computed. The argument
# and return type is dictionary from the feature/message name to tensor values.
# We can trigger this function using out ``send`` API:

g.send(0, 1, message_func=send_source)

###############################################################################
# Here, the message is computed using the feature of node#0. The result message
# (on 0->1) is not returned but directly saved in ``DGLGraph`` for the later
# receive phase.
#
# You can send multiple messages at once using the
# :ref:`multi-edge semantics <note-edge-broadcast>`.
# In such case, the source node and edge features are batched on the first dimension.
# You can simply print out the shape of the feature tensor in your message
# function.

def send_source_print(src, edge):
    print('src feat shape:', src['feat'].shape)
    return {'msg' : src['feat']}
g.send(0, [4, 5, 6], message_func=send_source_print)

###############################################################################
# To receive and aggregate in-coming messages, user can define a reduce function
# that operators on **a batch of nodes**.

def simple_reduce(node, msgs):
    return {'feat' : th.sum(msgs['msg'], dim=1)}

###############################################################################
# The reduce function has two arguments: ``node`` for the node features and
# ``msgs`` for the in-coming messages. It returns the updated node features.
# The function can be triggered using the ``recv`` API. Again, DGL support
# receive messages for multiple nodes at the same time. In such case, the
# node features are batched on the first dimension. Because each node can
# receive different number of in-coming messages, we divide the receiving
# nodes into buckets based on their numbers of receiving messages. As a result,
# the message tensor has at least three dimensions (B, n, D), where the second
# dimension concats all the messages for each node together. This also means
# the reduce UDF will be called for each bucket. You can simply print out
# the shape of the message tensor as follows:

def simple_reduce_print(node, msgs):
    print('msg shape:', msgs['msg'].shape)
    return {'feat' : th.sum(msgs['msg'], dim=1)}
g.recv([1, 4, 5, 6], reduce_func=simple_reduce_print)
print(g.get_n_repr()['feat'])

###############################################################################
# You can see that, after send and recv, the value of node#0 has been propagated
# to node 1, 4, 5 and 6.


###############################################################################
# DGL message passing APIs
# ------------------------
#
# TODO(minjie): enable backreference for all the mentioned APIs below.
#
# In DGL, we categorize the message passing APIs into three levels. All of them
# can be configured using UDFs such as the message and reduce functions.
#
# **Level-1 routines:** APIs that trigger computation on either a batch of nodes
# or a batch of edges. This includes:
#
# * ``send(u, v)`` and ``recv(v)``
# * ``update_edge(u, v)``: This updates the edge features using the current edge
#   features and the source and destination nodes features.
# * ``apply_nodes(v)``: This transforms the node features using the current node
#   features.
# * ``apply_edges(u, v)``: This transforms the edge features using the current edge
#   features.


###############################################################################
# **Level-2 routines:** APIs that combines several level-1 routines.
# 
# * ``send_and_recv(u, v)``: This first computes messages over u->v, then reduce
#   them on v. An optional node apply function can be provided.
# * ``pull(v)``: This computes the messages over all the in-edges of v, then reduce
#   them on v. An optional node apply function can be provided.
# * ``push(v)``: This computes the messages over all the out-edges of v, then
#   reduce them on the successors. An optional node apply function can be provided.
# * ``update_all()``: Send out and reduce messages on every node. An optional node
#   apply function can be provided.
#
# The following example uses ``send_and_recv`` to continue propagate signals to the
# sink node#9:

g.send_and_recv([1, 4, 5, 6], 9, message_func=send_source, reduce_func=simple_reduce)
print(g.get_n_repr()['feat'])

###############################################################################
# **Level-3 routines:** APIs that calls multiple level-2 routines.
#
# * ``propagate()``: TBD after Yu's traversal PR.

###############################################################################
# Builtin functions
# -----------------
#
# Since many message and reduce UDFs are very common (such as sending source
# node features as the message and aggregating messages using summation), DGL
# actually provides builtin functions that can be directly used:

import dgl.function as fn
g.send_and_recv(0, [2, 3], fn.copy_src(src='feat', out='msg'), fn.sum(msg='msg', out='feat'))
print(g.get_n_repr()['feat'])

###############################################################################
# TODO(minjie): document on multiple builtin function syntax after Lingfan
# finished his change.

###############################################################################
# Using builtin functions not only saves your time in writing codes, but also
# allows DGL to use more efficient implementation automatically. To see this,
# you can continue to our tutorial on Graph Convolutional Network.
# TODO(minjie): need a hyperref to the GCN tutorial here.
