"""
Use DGLGraph
============

In this tutorial, we introduce how to use our graph class -- ``DGLGraph``.
The ``DGLGraph`` is the very core data structure in our library. It provides the basic
interfaces to manipulate graph structure, set/get node/edge features and convert
from/to many other graph formats. You can also perform computation on the graph
using our message passing APIs. (TODO: give a link here to the message passing doc)
"""

###############################################################################
# Construct a graph
# -----------------
# 
# In ``DGLGraph``, all nodes are represented using consecutive integers starting from
# zero. All edges are directed. Let us start by creating a star network of 10 nodes
# where all the edges point to the center node (node#0).
# TODO(minjie): it's better to plot the graph here.

import dgl
star = dgl.DGLGraph()
star.add_nodes(10)  # add 10 nodes
for i in range(1, 10):
    star.add_edge(i, 0)
print('#Nodes:', star.number_of_nodes())
print('#Edges:', star.number_of_edges())


###############################################################################
# ``DGLGraph`` also supports adding multiple edges at once by providing multiple
# source and destination nodes. Multiple nodes are represented using either a
# list or a 1D integer tensor(vector). In addition to this, we also support
# "edge broadcasting":
#
# .. note::
# 
#   Given two source and destination node list/tensor ``u`` and ``v``.
#
#   - If ``len(u) == len(v)``, then this is a many-many edge set and
#     each edge is represented by ``(u[i], v[i])``.
#   - If ``len(u) == 1``, then this is a one-many edge set.
#   - If ``len(v) == 1``, then this is a many-one edge set.
#
# Edge broadcasting is supported in many APIs whenever a bunch of edges need
# to be specified. The example below creates the same star graph as the previous one.

star.clear()  # clear the previous graph
star.add_nodes(10)
u = list(range(1, 10))  # can also use tensor type here (e.g. torch.Tensor)
star.add_edges(u, 0)  # many-one edge set
print('#Nodes:', star.number_of_nodes())
print('#Edges:', star.number_of_edges())


###############################################################################
# In ``DGLGraph``, each edge is assigned an internal edge id (also a consecutive
# integer starting from zero). The ids follow the addition order of the edges
# and you can query the id using the ``edge_ids`` interface.

print(star.edge_ids(1, 0))  # the first edge
print(star.edge_ids([8, 9], 0))  # ask for ids of multiple edges


###############################################################################
# Assigning consecutive integer ids for nodes and edges makes it easier to batch
# their features together (see next section). As a result, removing nodes or edges
# of a ``DGLGraph`` is currently not supported because this will break the assumption
# that the ids form a consecutive range from zero.


###############################################################################
# Node and edge features
# ----------------------
# Nodes and edges can have feature data in tensor type. They can be accessed/updated
# through a key-value storage interface. The key must be hashable. The value should
# be features of each node and edge batched on the *first* dimension. For example,
# following codes create features for all nodes (``hv``) and features for all
# edges (``he``). Each feature is a vector of length 3.
#
# .. note::
#
#   The first dimension is usually reserved as batch dimension in DGL. Thus, even setting
#   only one node/edge still needs to have an extra dimension (of length one).

import torch as th
D = 3  # the feature dimension
N = star.number_of_nodes()
M = star.number_of_edges()
nfeat = th.randn((N, D))  # some random node features
efeat = th.randn((M, D))  # some random edge features
# TODO(minjie): enable following syntax
# star.nodes[:]['hv'] = nfeat
# star.edges[:]['he'] = efeat
star.set_n_repr({'hv' : nfeat})
star.set_e_repr({'he' : efeat})


###############################################################################
# We can then set some nodes' features to be zero.

# TODO(minjie): enable following syntax
# print(star.nodes[:]['hv'])
print(star.get_n_repr()['hv'])
# set node 0, 2, 4 feature to zero
star.set_n_repr({'hv' : th.zeros((3, D))}, [0, 2, 4])
print(star.get_n_repr()['hv'])


###############################################################################
# Once created, each node/edge feature will be associated with a *scheme* containing
# the shape, dtype information of the feature tensor. Updating features using data
# of different scheme will raise error unless all the features are updated,
# in which case the scheme will be replaced with the new one.

print(star.node_attr_schemes())
# updating features with different scheme will raise error
# star.set_n_repr({'hv' : th.zeros((3, 2*D))}, [0, 2, 4])
# updating all the nodes is fine, the old scheme will be replaced
star.set_n_repr({'hv' : th.zeros((N, 2*D))})
print(star.node_attr_schemes())


###############################################################################
# If a new feature is added for some but not all of the nodes/edges, we will
# automatically create empty features for the others to make sure that features are
# always aligned. By default, we fill zero for the empty features. The behavior
# can be changed using ``set_n_initializer`` and ``set_e_initializer``.

star.set_n_repr({'hv_1' : th.randn((3, D+1))}, [0, 2, 4])
print(star.node_attr_schemes())
print(star.get_n_repr()['hv_1'])


###############################################################################
# Convert from/to other formats
# -----------------------------
#



###############################################################################
# Multi-edge graph
# ----------------
#
