"""
.. currentmodule:: dgl

DGL at a glance
=========================

**Author**: `Minjie Wang <https://jermainewang.github.io/>`_, Quan Gan, `Jake
Zhao <https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang

The goal of this tutorial:

- Understand how DGL builds a graph from a high level.
- Perform simple computation on graphs.

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
# *This tutorial assumes basic familiarity with networkx.*

###############################################################################
# Building a graph
# ----------------
#
# A graph is built using :class:`~dgl.DGLGraph` class.
# Here as a toy example, we define a toy graph with two nodes then assign
# features on nodes and edges:

import torch as th
import networkx as nx
import dgl

def a_boring_graph():
    g = dgl.DGLGraph()
    g.add_nodes(2)
    g.add_edge(1, 0)

    # node and edge features
    x = th.tensor([[0.0, 0.0], [1.0, 2.0]])
    w = th.tensor([2]).float()
    g.ndata['x'] = x
    g.edata['w'] = w

    return g

###############################################################################
# We can also convert a graph defined by `networkx
# <https://networkx.github.io/documentation/stable/>`_ to DGL:

def an_interesting_graph():
    import networkx as nx

    N = 70
    g = nx.erdos_renyi_graph(N, 0.1)
    g = dgl.DGLGraph(g)

    x = th.randn(N, 6)
    w = th.randn(g.number_of_edges(), 1)
    g.ndata['x'] = x
    g.edata['w'] = w

    return g

###############################################################################
#  By default, DGLGraph object is directional:

g_boring = a_boring_graph()
g_better = an_interesting_graph()

import matplotlib.pyplot as plt
nx.draw(g_better.to_networkx(), node_size=50, node_color=[[.5, .5, .5,]])
plt.show()

###############################################################################
# Define Computation
# ------------------
# The canonical functionality of DGL is to provide efficient message passing
# and merging on graphs. It is implemented by using a message passing interface
# powered by the scatter-gather paradigm (i.e. a mailbox metaphor).
#
# To give an intuitive example, suppose we have one node :math:`v` , together with
# many incoming edges: :math:`e_i\in\mathcal{N}(v)`. Each node and edge is
# tagged with their own feature. Now, we can perform one iteration of message
# passing and merging by the following routine:
#
# - Each edge :math:`e_i` passes the information along into the node :math:`v`, by
#   ``send_source``.
# - A ``reduce`` operation is triggered to gather these messages
#   sent from the edges, by ``simple_reduce``.
# - ``readout`` function is called eventually to yield the updated feature on
#   :math:`v`.
#
# A graphical demonstration is displayed below, followed by a complete
# implementation.
#
# .. image:: https://drive.google.com/uc?export=view&id=1rc9cR0Iw96m_wjS55V9LJOJ4RpQBja15
#    :height: 300px
#    :width: 400px
#    :alt: mailbox
#    :align: center
#

def super_useful_comp(g):

    def send_source(edges):
        return {'msg': edges.src['x'] * edges.data['w']}

    def simple_reduce(nodes):
        msgs = nodes.mailbox['msg']
        return {'x': msgs.sum(1) + nodes.data['x']}

    def readout(g):
        return th.sum(g.ndata['x'], dim=0)

    g.register_message_func(send_source)
    g.register_reduce_func(simple_reduce)

    g.send(g.edges())
    g.recv(g.nodes())

    return readout(g)

###############################################################################
# See the python wrapper:

g_boring = a_boring_graph()
graph_sum = super_useful_comp(g_boring)
print("graph sum is: ", graph_sum)

g_better = an_interesting_graph()
graph_sum = super_useful_comp(g_better)
print("graph sum is: ", graph_sum)

###############################################################################
# Next steps
# ----------
# In the :doc:`next tutorial <2_basics>`, we will go through some more basics
# of DGL, such as reading and writing node/edge features.
