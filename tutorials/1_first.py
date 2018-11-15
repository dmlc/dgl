"""
.. currentmodule:: dgl

DGL at a glance
=========================

**Author**: `Minjie Wang <https://jermainewang.github.io/>`_, Quan Gan, `Jake
Zhao <https://cs.nyu.edu/~jakezhao/>`_, Zheng Zhang

The goal of DGL is to build, train, and deploy *machine learning models*
on *graph-structured data*.  To achieve this, DGL provides a :class:`DGLGraph`
class that defines the graph structure and the information on its nodes
and edges.  It also provides a set of feature transformation methods
and message passing methods to propagate information between nodes and edges.

Goal of this tutorial: get a feeling of how DGL looks like!
"""

###############################################################################
# Building a graph
# ----------------
# Let's build a toy graph with two nodes and throw some representations on the
# nodes and edges:

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
# We can also convert from networkx:

def an_interesting_graph():
    import networkx as nx

    N = 100
    g = nx.erdos_renyi_graph(N, 0.1)
    g = dgl.DGLGraph(g)

    x = th.randn(N, 6)
    w = th.randn(g.number_of_edges(), 1)
    g.ndata['x'] = x
    g.edata['w'] = w

    return g

###############################################################################
# One thing to be aware of is that :class:`DGLGraph` is directional:

g_boring = a_boring_graph()
g_better = an_interesting_graph()

import matplotlib.pyplot as plt
nx.draw(g_better.to_networkx(), node_size=50, node_color=[[.5, .5, .5,]])
plt.show()

###############################################################################
# Define Computation
# ------------------
# The focus of DGL is to provide a way to integrate representation learning
# (using neural networks) with graph data. The way we do it is with a
# message-passing interface with scatter-gather paradigm. (i.e. a mailbox metaphor).
#
# .. note::
#
#    For people familiar with graph convolutional network, it is easy to see the
#    pattern here.

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
# The point is, regardless of what kind of graphs and the form of representations,
# DGL handles it uniformly and efficiently.

g_boring = a_boring_graph()
graph_sum = super_useful_comp(g_boring)
print("graph sum is: ", graph_sum)

g_better = an_interesting_graph()
graph_sum = super_useful_comp(g_better)
print("graph sum is: ", graph_sum)

###############################################################################
# Next steps
# ----------
# In the :doc:`next tutorial <2_basics>`, we will go through defining
# a graph structure, as well as reading and writing node/edge representations.
