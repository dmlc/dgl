

"""
Graph Convolutional Network New
====================================
**Author**: `Quan Gan`

In this tutorial, we will go through the basics of DGL, in the following order:
    1. Creating a graph
    2. Setting/getting node/edge states
    3. Updating node/edge states using user-defined functions
    4. Passing information from node to adjacent edges, and edges to adjacent
       nodes
    5. Implementing a Graph Convolutional Network (GCN) and a Graph Attention
       Network (GAT)
    6. Using built-in functions to simplify your implementation
"""

##############################################################################
# Section 1. Creating a Graph
# ---------------------------
#
# Let's say we want to create the following graph:
#
# .. digraph:: foo
#
#    digraph foo {
#            layout=circo;
#            "A" -> "B" -> "C" -> "A";
#    }
#
# First, we need to create a ``DGLGraph`` object.

from dgl import DGLGraph

g = DGLGraph()


##############################################################################
# And then we add 3 vertices (or *nodes*) into ``g``:

g.add_nodes(3)


##############################################################################
# In DGL, all vertices are uniquely identified by integers, starting from 0.
# Assuming that we map the node ``A``, ``B``, and ``C`` to ID 0, 1, and 2, we
# can add the edges of the desired graph above as follows:

g.add_edge(0, 1)
g.add_edge(1, 2)
g.add_edge(2, 0)
# Or, equivalently
# g.add_edges([0, 1, 2], [1, 2, 0])


##############################################################################
# All the edges are also uniquely identified by integers, again starting from
# 0.  The edges are labeled in the order of addition.  In the example above,
# the edge ``0 -> 1`` is labeled as edge #0, ``1 -> 2`` as edge #1, and
# ``2 -> 0`` as edge #2.


##############################################################################
# Section 2. Setting/getting node/edge states
# --------------------------------------
# Now, we wish to assign the nodes some states, or features.
#
# In DGL, the node/edge states are represented as dictionaries, with strings
# as keys (or *fields*), and tensors as values.  DGL aims to be
# framework-agnostic, and currently it supports PyTorch and MXNet.  From now
# on, we use PyTorch as an example.
#
# You can set up states for some or all nodes at the same time in DGL.
# All you need is to stack the tensors along the first dimension for each
# key, and feed the dictionary of the stacked tensors into ``set_n_repr``
# as a whole.

import torch

# We are going to assign each node two states X and Y.  For each node,
# X is a 2-D vector and Y is a 2x4 matrix.  You only need to make sure
# the tensors with the same key across all the (set) nodes to have the
# same shape and data type.
X = torch.randn(3, 2)
Y = torch.randn(3, 2, 4)

# You can set the states for all of them...
g.set_n_repr({'X': X, 'Y': Y})
# ... or setting partial states, but only after you have set all nodes on
# at least one key.
# TODO: do we want to fix this behavior to allow initial partial setting?
g.set_n_repr({'X': X[0:2], 'Y': Y[0:2]}, [0, 1])


##############################################################################
# You can also efficiently get the node states as a dictionary of tensors.
# The dictionary will also have strings as keys and stacked tensors as values.

# Getting all node states.  The tensors will be stacked along the first
# dimension, in the same order as node ID.
n_repr = g.get_n_repr()
X_ = n_repr['X']
Y_ = n_repr['Y']
assert torch.allclose(X_, X)
assert torch.allclose(Y_, Y)

# You can also get the states from a subset of nodes.  The tensors will be
# stacked along the first dimension, in the same order as what you feed in.
n_repr_subset = g.get_n_repr([0, 2])
X_ = n_repr_subset['X']
Y_ = n_repr_subset['Y']
assert torch.allclose(X_, X[[0, 2]])
assert torch.allclose(Y_, Y[[0, 2]])


##############################################################################
# Setting/getting edge states is very similar.  We provide two ways of reading
# and writing edge states: by source-destination pairs, and by edge ID.

# We are going to assign each edge a state A, which is a 3-D vector for each
# edge.
A = torch.randn(3, 3)

# You can either set the states of all edges...
g.set_e_repr({'A': A})
# ... or by source-destination pair (in this case, assigning A[0] to (0 -> 1)
# and A[2] to (2 -> 0) ...
g.set_e_repr({'A': A[[0, 2]]}, [0, 2], [1, 0])
# ... or by edge ID (#0 and #2)
g.set_e_repr_by_id({'A': A[[0, 2]]}, [0, 2])
# Note that the latter two options are available only if you have set at least
# one field on all edges.
# TODO: do we want to fix this behavior to allow initial partial setting?

# Getting edge states is also easy...
e_repr = g.get_e_repr()
A_ = e_repr['A']
assert torch.allclose(A_, A)
# ... and you can also do it either by specifying source-destination pair...
e_repr_subset = g.get_e_repr([0], [1])
assert torch.allclose(e_repr_subset['A'], A[[0]])
# ... or by edge ID
e_repr_subset = g.get_e_repr_by_id([0])
assert torch.allclose(e_repr_subset['A'], A[[0]])


##############################################################################
# One can also remove node/edge states from the graph.  This is particularly
# useful to save memory during inference.

A_ = g.pop_e_repr('A')
assert torch.allclose(A_, A)
