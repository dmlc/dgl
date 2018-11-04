"""
.. _tutorial-graph:

Use DGLGraph
============
**Author**: `Minjie Wang <https://jermainewang.github.io/>`_

In this tutorial, we introduce how to use our graph class -- ``DGLGraph``.
The ``DGLGraph`` is the very core data structure in our library. It provides the basic
interfaces to manipulate graph structure, set/get node/edge features and convert
from/to many other graph formats. You can also perform computation on the graph
using our message passing APIs (see :ref:`tutorial-mp`).

TODO: 1) explain `tensor`; 2) enable g.nodes/edges[:][key]; 3) networkx conversion in one place
"""

###############################################################################
# Construct a graph
# -----------------
#
# The design of ``DGLGraph`` was influenced by other graph libraries. Indeed, you can
# create a graph from `networkx <https://networkx.github.io/>`__, and convert it into a ``DGLGraph``
# and vice versa:

import networkx as nx
import dgl

g_nx = nx.petersen_graph()
g_dgl = dgl.DGLGraph(g_nx)

import matplotlib.pyplot as plt
plt.subplot(121)
nx.draw(g_nx, with_labels=True)
plt.subplot(122)
nx.draw(g_dgl.to_networkx(), with_labels=True)

plt.show()

###############################################################################
# They are the same graph, except that ``DGLGraph`` are always `directional`.
#
# Creating a graph is a matter of specifying total number of nodes and the edges among them.
# In ``DGLGraph``, all nodes are represented using consecutive integers starting from
# zero, and you can add more nodes repeatedly.
#
# .. note::
#
#  ``nx.add_node(100)`` adds a node with id 100, ``dgl.add_nodes(100)`` adds another 100 nodes into the graph.

g_dgl.clear()
g_nx.clear()
g_dgl.add_nodes(20)
print("We have %d nodes now" % g_dgl.number_of_nodes())
g_dgl.add_nodes(100)
print("Now we have %d nodes!" % g_dgl.number_of_nodes())
g_nx.add_node(100)
print("My nx buddy only has %d :( " % g_nx.number_of_nodes())

###############################################################################
# The most naive way to add edges are just adding them one by one, with a (*src, dst*) pair.
# Let's generate a star graph where all the edges point to the center (node#0).

star = dgl.DGLGraph()
star.add_nodes(10)  # add 10 nodes
for i in range(1, 10):
    star.add_edge(i, 0)
nx.draw(star.to_networkx(), with_labels=True)

###############################################################################
# It's more efficient to add many edges with a pair of list, or better still, with a pair of tensors.
# TODO: needs to explain ``tensor``, since it's not a Python primitive data type.

# using lists
star.clear()
star.add_nodes(10)
src = [i for i in range(1, 10)]; dst = [0]*9
star.add_edges(src, dst)

# using tensor
star.clear()
star.add_nodes(10)
import torch as th
src = th.tensor(src); dst = th.tensor(dst)
star.add_edges(src, dst)

###############################################################################
# In addition to this, we also support
# "edge broadcasting":
#
# .. _note-edge-broadcast:
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

###############################################################################
# In ``DGLGraph``, each edge is assigned an internal edge id (also a consecutive
# integer starting from zero). The ids follow the addition order of the edges
# and you can query the id using the ``edge_ids`` interface, which returns a tensor.

print(star.edge_ids(1, 0))  # query edge id of 1->0; it happens to be the first edge!
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
# be features of each node and edge, batched on the *first* dimension. For example,
# the following codes create features for all nodes (``hv``) and features for all
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
# .. note::
#    The first dimension of the node feature has length equal the number of nodes,
#    whereas of the edge feature the number of edges.
#
# We can then set some nodes' features to be zero.

# TODO(minjie): enable following syntax
# print(star.nodes[:]['hv'])
print("node features:")
print(star.get_n_repr()['hv'])
print("\nedge features:")
print(star.get_e_repr()['he'])
# set node 0, 2, 4 feature to zero
print("\nresetting features at node 0, 2 and 4...")
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
# always aligned. By default, we zero-fill the empty features. The behavior
# can be changed using ``set_n_initializer`` and ``set_e_initializer``.

star.set_n_repr({'hv_1' : th.randn((3, D+1))}, [0, 2, 4])
print(star.node_attr_schemes())
print(star.get_n_repr()['hv'])
print(star.get_n_repr()['hv_1'])


###############################################################################
# Convert from/to other formats
# -----------------------------
# DGLGraph can be easily converted from/to ``networkx`` graph.

import networkx as nx
# note that networkx create undirected graph by default, so when converting
# to DGLGraph, directed edges of both directions will be added.
nx_star = nx.star_graph(9)
star = dgl.DGLGraph(nx_star)
print('#Nodes:', star.number_of_nodes())
print('#Edges:', star.number_of_edges())


###############################################################################
# Node and edge attributes can be automatically batched when converting from
# ``networkx`` graph. Since ``networkx`` graph by default does not tell which
# edge is added the first, we use the ``"id"`` edge attribute as a hint
# if available.

for i in range(10):
    nx_star.nodes[i]['feat'] = th.randn((D,))
star = dgl.DGLGraph()
star.from_networkx(nx_star, node_attrs=['feat'])  # auto-batch specified node features
print(star.get_n_repr()['feat'])


###############################################################################
# Multi-edge graph
# ----------------
# There are many applications that work on graphs containing multi-edges. To enable
# this, construct ``DGLGraph`` with ``multigraph=True``.

g = dgl.DGLGraph(multigraph=True)
g.add_nodes(5)
g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(0, 1)
print('#Nodes:', g.number_of_nodes())
print('#Edges:', g.number_of_edges())
# init random edge features
M = g.number_of_edges()
g.set_e_repr({'he' : th.randn((M, D))})


###############################################################################
# Because an edge in multi-graph cannot be uniquely identified using its incident
# nodes ``u`` and ``v``, you need to use edge id to access edge features. The
# edge ids can be queried from ``edge_id`` interface.

eid_01 = g.edge_id(0, 1)
print(eid_01)


###############################################################################
# We can then use the edge id to set/get the features of the corresponding edge.
g.set_e_repr_by_id({'he' : th.ones(len(eid_01), D)}, eid=eid_01)
print(g.get_e_repr()['he'])
