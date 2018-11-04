"""
.. _tutorial-first:
Getting started with DGL
========================
**Author**: `Minjie Wang <https://jermainewang.github.io/>`_, `Quan Gan`, `Yu Gai`, `Qi Huang`, `Zheng Zhang`

The goal of DGL is to build, train, and deploy *machine learning models* on *graph-structured data*.  To achieve this, DGL provides a `DGLGraph` class that defines the graph structure and the information on its nodes and edges.  It also provides a set of feature transformation methods and message passing methods to propagate information between nodes and edges.

The design of `DGLGraph` was influenced by other graph libraries. Indeed, you can create a graph from networkx, and convert it into a `DGLGraph` and vice versa:
"""

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
#################################################
# They are the same graph, except that DGLGraph are *always* directional.
#
# DGLGraph basics: nodes, edges and features
# -----------------------------------------------
# One can also assign features to nodes and edges of a `DGLGraph`.  The features are represented as dictionary of names (strings) and tensors.
#
# .. note::
#    DGL aims to be framework-agnostic, and currently it supports PyTorch and MXNet tensors. From now on, we use PyTorch as an example.
#
import torch as th

g = dgl.DGLGraph()
g.add_nodes(10)
x = th.randn(10, 3)
g.ndata['x'] = x

#######################
# `ndata` is a syntax sugar to access states of all nodes, states are stored in a container `data` that hosts user defined dictionary.
#
print(g.ndata['x'] == g.nodes[:].data['x'])
############################################
# access node set in a variety of way: integer, list, integer tensor
g.nodes[0].data['x'] = th.zeros(1, 3)
g.nodes[[0, 1, 2]].data['x'] = th.zeros(3, 3)
g.nodes[th.tensor([0, 1, 2])].data['x'] = th.zeros(3, 3)
##############################################################
# Let's build a star graph. DGLGraph nodes are consecutive range of integers between 0 and ``g.number_of_nodes()`` and can grow by calling ``g.add_nodes``. DGLGraph edges are in order of their additions. Note that edges are accessed in much the same way as nodes, with one extra feature of *edge broadcasting*:
#
star = dgl.DGLGraph()
star.add_nodes(10)
# a couple edges one-by-one
for i in range(1, 4):
    star.add_edge(i, 0)
# a few more with a paired list
src = list(range(5, 8)); dst = [0]*3
star.add_edges(src, dst)
# finish with a pair of tensors
src = th.tensor([8, 9]); dst = th.tensor([0, 0])
star.add_edges(src, dst)

# edge broadcasting will do star graph in one go!
star.clear(); star.add_nodes(10)
src = th.tensor(list(range(1, 10)));
star.add_edges(src, 0)

nx.draw(star.to_networkx(), with_labels=True)
plt.show()
##########################################################
# Multigraph
# ```````````````````
# Many graph applications need multi-edges. To enable this, construct DGLGraph with ``multigraph=True``.
#
star = dgl.DGLGraph(multigraph=True)
star.add_nodes(10)
star.ndata['x'] = th.randn(10, 2)

star.add_edges(list(range(1, 10)), 0)
star.add_edge(1, 0) # two edges on 0->1

# edges can have features, too!
star.edata['w'] = th.randn(10, 2)
star.edges[1].data['w'] = th.zeros(1, 2)
print(star.edges())
##################################################
# Edge ids
# ```````````````
# An edge in multi-graph cannot be uniquely identified using its incident nodes :math:`u` and :math:`v`; query their edge ids use ``edge_id`` interface.
#
eid_10 = star.edge_id(1, 0)
star.edges[eid_10].data['w'] = th.ones(len(eid_10), 2)
print(star.edata['w'])
############################################################
# Node/edge schemes
# ```````````````````````
# Once created, each node/edge field will be associated with a **scheme** containing the *shape* and *data type (dtype)* of its field value.
#
print(star.node_attr_schemes())
star.ndata['x'] = th.zeros((10, 4))
print(star.node_attr_schemes())
#########################################
# One can also remove node/edge states from the graph. This is particularly useful to save memory during inference,

del star.ndata['x']
del star.edata['w']
###########################
#
# .. note::
#    * Nodes and edges can be added but not removed; we will support removal in the future.
#    * Updating a feature of different schemes raise error on indivdual node (or node subset).
#
# DGL Example: PageRank
# --------------------------
# In this section we illustrate the usage of different levels of message passing API with PageRank on a small graph. In DGL, the message passing and feature transformations are all **User-Defined Functions** (UDFs).
#
# The PageRank Algorithm
# ```````````````````````````
# In each iteration of PageRank, every node (web page) first scatters its PageRank value uniformly to its downstream nodes. The new PageRank value of each node is computed by aggregating the received PageRank values from its neighbors, which is then adjusted by the damping factor:
# :math:`PV(u) = \frac{1-d}{N} + d \times \sum_{v \in \mathcal{N}(u)} \frac{PV(v)}{D(v)}`
# , where :math:`N` is the number of nodes in the graph; :math:`D(v)` is the out-degree of a node :math:`v`; and :math:`\mathcal{N}(u)` is the neighbor nodes.
#
# A naive implementation
# ```````````````````````````
# Let us first create a graph with 100 nodes with NetworkX and convert it to a ``DGLGraph``:
#

N = 100  # number of nodes
DAMP = 0.85  # damping factor
K = 10  # number of iterations
g = nx.erdos_renyi_graph(N, 0.05)
g = dgl.DGLGraph(g)
nx.draw(g.to_networkx(), node_size=100)
plt.show()
########################################
# According to the algorithm, PageRank consists of two phases in a typical scatter-gather pattern. We first initialize the PageRank value of each node to :math:`\frac1N` and store each node's out-degree as a node feature:
#
g.ndata['pv'] = th.ones(N) / N
g.ndata['deg'] = g.out_degrees(g.nodes()).float()
###################################################
# We then define the message function, which divides every node's PageRank value by its out-degree and passes the result as message to its neighbors:
#
def pagerank_message_func(edges):
    return {'pv' : edges.src['pv'] / edges.src['deg']}
###########################################################
# In DGL, the message functions are expressed as **Edge UDFs**.  Edge UDFs take in a single argument `edges`.  It has three members `src`, `dst`, and `data` for accessing source node features, destination node features, and edge features respectively.  Here, the function computes messages only from source node features.

# Next, we define the reduce function, which removes and aggregates the messages from its ``mailbox``, and computes its new PageRank value:
#
def pagerank_reduce_func(nodes):
    msgs = th.sum(nodes.mailbox['pv'], dim=1)
    pv = (1 - DAMP) / N + DAMP * msgs
    return {'pv' : pv}
##################################################
# The reduce functions are **Node UDFs**.  Node UDFs have a single argument `nodes`, which has two members `data` and `mailbox`.  `data` contains the node features while `mailbox` contains all incoming message features, stacked along the second dimension (hence the `dim=1` argument).
# The message UDF works on a batch of edges, whereas the reduce  UDF works on a batch of edges but outputs a batch of nodes. Their relationships are as follows,
#
# .. image:: https://i.imgur.com/kIMiuFb.png
# We register the message function and reduce function, which will be called later by DGL.
#
g.register_message_func(pagerank_message_func)
g.register_reduce_func(pagerank_reduce_func)
####################################################
def pagerank_naive(g):
    # Phase #1: send out messages along all edges.
    for u, v in zip(*g.edges()):
        g.send((u, v))
    # Phase #2: receive messages to compute new PageRank values.
    for v in g.nodes():
        g.recv(v)
#######################################################################
# Improvement with batch semantics
# `````````````````````````````````````
# The above code does not scale to large graph because it iterates over all the nodes. DGL solves this by letting user compute on a *batch* of nodes or edges. For example, the following codes trigger message and reduce functions on multiple nodes and edges at once.
#
def pagerank_batch(g):
    g.send(g.edges())
    g.recv(g.nodes())
###########################
# Note that we are still using the same reduce function `pagerank_reduce_func`, where `nodes.mailbox['pv']` is a *single* tensor, stacking the incoming messages along the second dimension.
# Naturally, one will wonder if this is even possible to perform reduce on all nodes in parallel, since each node may have different number of incoming messages and one cannot really "stack" tensors of different lengths together.  In general, DGL solves the problem by grouping the nodes by the number of incoming messages, and calling the reduce function for each group.
#
# More improvement with higehr level APIs
# ```````````````````````````````````````````````
# DGL provides many routines that combines basic `send` and `recv` in various ways. They are called **level-2 APIs**. For example, the PageRank example can be further simplified as follows:
#
def pagerank_level2(g):
    g.update_all()
###########################
# Besides `update_all`, we also have `pull`, `push`, and `send_and_recv` in this level-2 category. Please refer to their own API reference documents for more details. (TODO: a link to the document).
#
# Further improvement with DGL built-in functions
# `````````````````````````````````````````````````
# As some of the message and reduce functions are very commonly used, DGL also provides **builtin functions**. For example, two builtin functions can be used in the PageRank example.
#
# * `dgl.function.copy_src(src, out)` is an edge UDF that computes the output using the source node feature data. User needs to specify the name of the source feature data (`src`) and the output name (`out`).
#
# * `dgl.function.sum(msg, out)` is a node UDF that sums the messages in the node's mailbox. User needs to specify the message name (`msg`) and the output name (`out`).
# For example, the PageRank example can be rewritten as following:
#
import dgl.function as fn

def pagerank_builtin(g):
    g.ndata['pv'] = g.ndata['pv'] / g.ndata['deg']
    g.update_all(message_func=fn.copy_src(src='pv', out='m'),
                 reduce_func=fn.sum(msg='m',out='m_sum'))
    g.ndata['pv'] = (1 - DAMP) / N + DAMP * g.ndata['m_sum']
#########################################################################
# Here, we directly provide the UDFs to the `update_all` as its arguments. This will override the previously registered UDFs.
# Check ([here] \TODO: add link) to understand why spMV can speed up the scatter-gather phase in PageRank. For more details about the builtin functions in DGL, please read their API reference documents. (TODO: a link here).
# You can also download and run the codes to feel the difference.
for k in range(K):
    # Uncomment the corresponding line to select different version.
    # pagerank_naive(g)
    # pagerank_batch(g)
    # pagerank_level2(g)
    pagerank_builtin(g)
print(g.ndata['pv'])
#########################################################################
# Using spMV for PageRank
# ````````````````````````
# Using builtin functions allows DGL to understand the semantics of UDFs and thus allows more efficient implementation for you. For example, in the case of PageRank, one common trick to accelerate it is using its linear algebra form.
#
# :math:`\mathbf{R}^{k} = \frac{1-d}{N} \mathbf{1} + d \mathbf{A}*\mathbf{R}^{k-1}`
#
# Here, :math:`\mathbf{R}^k` is the vector of the PageRank values of all nodes at iteration :math:`k`; :math:`\mathbf{A}` is the sparse adjacency matrix of the graph. Computing this equation is quite efficient because there exists efficient GPU kernel for the *sparse-matrix-vector-multiplication* (spMV). DGL detects whether such optimization is available through the builtin functions. If the certain combination of builtins can be mapped to a spMV kernel (e.g. the pagerank example), DGL will use it automatically. As a result, *we recommend using builtin functions whenever it is possible*.



