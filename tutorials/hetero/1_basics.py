"""
.. currentmodule:: dgl

Working with Heterogeneous Graphs in DGL
========================================

**Author**: Quan Gan, `Minjie Wang <https://jermainewang.github.io/>`_, Mufei Li,
George Karypis, Zheng Zhang

In this tutorial, you learn about:

* Examples of heterogenous graph data and typical applications.

* Creating and manipulating a heterogenous graph in DGL.

* Implementing `Relational-GCN <https://arxiv.org/abs/1703.06103>`_, a popular GNN model,
  for heterogenous graph input.

* Training a model to solve a node classification task.

Heterogeneous graphs, or *heterographs* for short, are graphs that contain
different types of nodes and edges. The different types of nodes and edges tend
to have different types of attributes that are designed to capture the
characteristics of each node and edge type. Within the context of
graph neural networks, depending on their complexity, certain node and edge types
might need to be modeled with representations that have a different number of dimensions.

DGL supports graph neural network computations on such heterogeneous graphs, by
using the heterograph class and its associated API.

"""

###############################################################################
# Examples of heterographs
# -----------------------
# Many graph datasets represent relationships among various types of entities.
# This section provides an overview for several graph use-cases that show such relationships 
# and can have their data represented as heterographs.
#
# Citation graph 
# ~~~~~~~~~~~~~~~
# The Association for Computing Machinery publishes an `ACM dataset <https://aminer.org/citation>`_ that contains two
# million papers, their authors, publication venues, and the other papers
# that were cited. This information can be represented as a heterogeneous graph.
#
# The following diagram shows several entities in the ACM dataset and the relationships among them 
# (taken from `Shi et al., 2015 <https://arxiv.org/pdf/1511.04854.pdf>`_).
#
# .. figure:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/hetero/acm-example.png# 
# 
# This graph has three types of entities that correspond to papers, authors, and publication venues.
# It also contains three types of edges that connect the following:
#
# * Authors with papers corresponding to *written-by* relationships
#
# * Papers with publication venues corresponding to *published-in* relationships
#
# * Papers with other papers corresponding to *cited-by* relationships
#
#
# Recommender systems 
# ~~~~~~~~~~~~~~~~~~~~ 
# The datasets used in recommender systems often contain
# interactions between users and items. For example, the data could include the
# ratings that users have provided to movies. Such interactions can be modeled
# as heterographs.
#
# The nodes in these heterographs will have two types, *users* and *movies*. The edges
# will correspond to the user-movie interactions. Furthermore, if an interaction is
# marked with a rating, then each rating value could correspond to a different edge type.
# The following diagram shows an example of user-item interactions as a heterograph.
#
# .. figure:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/hetero/recsys-example.png
#
#
# Knowledge graph 
# ~~~~~~~~~~~~~~~~
# Knowledge graphs are inherently heterogenous. For example, in
# Wikidata, Barack Obama (item Q76) is an instance of a human, which could be viewed as
# the entity class, whose spouse (item P26) is Michelle Obama (item Q13133) and
# occupation (item P106) is politician (item Q82955). The relationships are shown in the following.
# diagram.
#
# .. figure:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/hetero/kg-example.png
#

###############################################################################
# Creating a heterograph in DGL
# -----------------------------
# You can create a heterograph in DGL using the :func:`dgl.heterograph` API.
# The argument to :func:`dgl.heterograph` is a dictionary. The keys are tuples
# in the form of ``(srctype, edgetype, dsttype)`` specifying the relation name
# and the two entity types it connects. Such tuples are called *canonical edge
# types*. The values are data to initialize the graph structures, that is, which
# nodes the edges actually connect.
#
# For instance, the following code creates the user-item interactions heterograph shown earlier.

# Each value of the dictionary is a list of edge tuples.
# Nodes are integer IDs starting from zero. Nodes IDs of different types have
# separate countings.
import dgl

ratings = dgl.heterograph(
    {('user', '+1', 'movie') : [(0, 0), (0, 1), (1, 0)],
     ('user', '-1', 'movie') : [(2, 1)]})

###############################################################################
# DGL supports creating a graph from a variety of data sources. The following
# code creates the same graph as the above.
#
# Creating from scipy matrix
import scipy.sparse as sp
plus1 = sp.coo_matrix(([1, 1, 1], ([0, 0, 1], [0, 1, 0])), shape=(3, 2))
minus1 = sp.coo_matrix(([1], ([2], [1])), shape=(3, 2))
ratings = dgl.heterograph(
    {('user', '+1', 'movie') : plus1,
     ('user', '-1', 'movie') : minus1})

# Creating from networkx graph
import networkx as nx
plus1 = nx.DiGraph()
plus1.add_nodes_from(['u0', 'u1', 'u2'], bipartite=0)
plus1.add_nodes_from(['m0', 'm1'], bipartite=1)
plus1.add_edges_from([('u0', 'm0'), ('u0', 'm1'), ('u1', 'm0')])
# To simplify the example, reuse the minus1 object.
# This also means that you could use different sources of graph data
# for different relationships.
ratings = dgl.heterograph(
    {('user', '+1', 'movie') : plus1,
     ('user', '-1', 'movie') : minus1})

# Creating from edge indices
ratings = dgl.heterograph(
    {('user', '+1', 'movie') : ([0, 0, 1], [0, 1, 0]),
     ('user', '-1', 'movie') : ([2], [1])})

###############################################################################
# Manipulating heterograph
# ------------------------
# You can create a more realistic heterograph using the ACM dataset. To do this, first 
# download the dataset as follows:

import scipy.io
import urllib.request

data_url = 'https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ACM.mat'
data_file_path = '/tmp/ACM.mat'

urllib.request.urlretrieve(data_url, data_file_path)
data = scipy.io.loadmat(data_file_path)
print(list(data.keys()))

###############################################################################
# The dataset stores node information by their types: ``P`` for paper, ``A``
# for author, ``C`` for conference, ``L`` for subject code, and so on. The relationships
# are stored as SciPy sparse matrix under key ``XvsY``, where ``X`` and ``Y``
# could be any of the node type code.
#
# The following code prints out some statistics about the paper-author relationships.

print(type(data['PvsA']))
print('#Papers:', data['PvsA'].shape[0])
print('#Authors:', data['PvsA'].shape[1])
print('#Links:', data['PvsA'].nnz)

###############################################################################
# Converting this SciPy matrix to a heterograph in DGL is straightforward.

pa_g = dgl.heterograph({('paper', 'written-by', 'author') : data['PvsA']})
# equivalent (shorter) API for creating heterograph with two node types:
pa_g = dgl.bipartite(data['PvsA'], 'paper', 'written-by', 'author')

###############################################################################
# You can easily print out the type names and other structural information.

print('Node types:', pa_g.ntypes)
print('Edge types:', pa_g.etypes)
print('Canonical edge types:', pa_g.canonical_etypes)

# Nodes and edges are assigned integer IDs starting from zero and each type has its own counting.
# To distinguish the nodes and edges of different types, specify the type name as the argument.
print(pa_g.number_of_nodes('paper'))
# Canonical edge type name can be shortened to only one edge type name if it is
# uniquely distinguishable.
print(pa_g.number_of_edges(('paper', 'written-by', 'author')))
print(pa_g.number_of_edges('written-by'))
print(pa_g.successors(1, etype='written-by'))  # get the authors that write paper #1

# Type name argument could be omitted whenever the behavior is unambiguous.
print(pa_g.number_of_edges())  # Only one edge type, the edge type argument could be omitted

###############################################################################
# A homogeneous graph is just a special case of a heterograph with only one type
# of node and edge. In this case, all the APIs are exactly the same as in
# :class:`DGLGraph`.

# Paper-citing-paper graph is a homogeneous graph
pp_g = dgl.heterograph({('paper', 'citing', 'paper') : data['PvsP']})
# equivalent (shorter) API for creating homogeneous graph
pp_g = dgl.graph(data['PvsP'], 'paper', 'cite')

# All the ntype and etype arguments could be omitted because the behavior is unambiguous.
print(pp_g.number_of_nodes())
print(pp_g.number_of_edges())
print(pp_g.successors(3))

###############################################################################
# Create a subset of the ACM graph using the paper-author, paper-paper, 
# and paper-subject relationships.  Meanwhile, also add the reverse
# relationship to prepare for the later sections.

G = dgl.heterograph({
        ('paper', 'written-by', 'author') : data['PvsA'],
        ('author', 'writing', 'paper') : data['PvsA'].transpose(),
        ('paper', 'citing', 'paper') : data['PvsP'],
        ('paper', 'cited', 'paper') : data['PvsP'].transpose(),
        ('paper', 'is-about', 'subject') : data['PvsL'],
        ('subject', 'has', 'paper') : data['PvsL'].transpose(),
    })

print(G)

###############################################################################
# **Metagraph** (or network schema) is a useful summary of a heterograph.
# Serving as a template for a heterograph, it tells how many types of objects
# exist in the network and where the possible links exist.
#
# DGL provides easy access to the metagraph, which could be visualized using
# external tools.

# Draw the metagraph using graphviz.
import pygraphviz as pgv
def plot_graph(nxg):
    ag = pgv.AGraph(strict=False, directed=True)
    for u, v, k in nxg.edges(keys=True):
        ag.add_edge(u, v, label=k)
    ag.layout('dot')
    ag.draw('graph.png')

plot_graph(G.metagraph)

###############################################################################
# Learning tasks associated with heterographs
# -------------------------------------------
# Some of the typical learning tasks that involve heterographs include:
#
# * *Node classification and regression* to predict the class of each node or
#   estimate a value associated with it.
#
# * *Link prediction* to predict if there is an edge of a certain
#   type between a pair of nodes, or predict which other nodes a particular
#   node is connected with (and optionally the edge types of such connections).
#
# * *Graph classification/regression* to assign an entire
#   heterograph into one of the target classes or to estimate a numerical
#   value associated with it.
#
# In this tutorial, we designed a simple example for the first task.
#
# A semi-supervised node classification example
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Our goal is to predict the publishing conference of a paper using the ACM
# academic graph we just created. To further simplify the task, we only focus
# on papers published in three conferences: *KDD*, *ICML*, and *VLDB*. All
# the other papers are not labeled, making it a semi-supervised setting.
#
# The following code extracts those papers from the raw dataset and prepares 
# the training, validation, testing split.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

pvc = data['PvsC'].tocsr()
# find all papers published in KDD, ICML, VLDB
c_selected = [0, 11, 13]  # KDD, ICML, VLDB
p_selected = pvc[:, c_selected].tocoo()
# generate labels
labels = pvc.indices
labels[labels == 11] = 1
labels[labels == 13] = 2
labels = torch.tensor(labels).long()

# generate train/val/test split
pid = p_selected.row
shuffle = np.random.permutation(pid)
train_idx = torch.tensor(shuffle[0:800]).long()
val_idx = torch.tensor(shuffle[800:900]).long()
test_idx = torch.tensor(shuffle[900:]).long()

###############################################################################
# Relational-GCN on heterograph
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We use `Relational-GCN <https://arxiv.org/abs/1703.06103>`_ to learn the
# representation of nodes in the graph. Its message-passing equation is as
# follows:
#
# .. math::
#
#    h_i^{(l+1)} = \sigma\left(\sum_{r\in \mathcal{R}}
#    \sum_{j\in\mathcal{N}_r(i)}W_r^{(l)}h_j^{(l)}\right)
#
# Breaking down the equation, you see that there are two parts in the
# computation.
#
# (i) Message computation and aggregation within each relation :math:`r`
#
# (ii) Reduction that merges the results from multiple relationships
#
# Following this intuition, perform message passing on a heterograph in
# two steps.
#
# (i) Per-edge-type message passing
#
# (ii) Type wise reduction

import dgl.function as fn

class HeteroRGCNLayer(nn.Module):
    def __init__(self, in_size, out_size, etypes):
        super(HeteroRGCNLayer, self).__init__()
        # W_r for each relation
        self.weight = nn.ModuleDict({
                name : nn.Linear(in_size, out_size) for name in etypes
            })

    def forward(self, G, feat_dict):
        # The input is a dictionary of node features for each type
        funcs = {}
        for srctype, etype, dsttype in G.canonical_etypes:
            # Compute W_r * h
            Wh = self.weight[etype](feat_dict[srctype])
            # Save it in graph for message passing
            G.nodes[srctype].data['Wh_%s' % etype] = Wh
            # Specify per-relation message passing functions: (message_func, reduce_func).
            # Note that the results are saved to the same destination feature 'h', which
            # hints the type wise reducer for aggregation.
            funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
        # Trigger message passing of multiple types.
        # The first argument is the message passing functions for each relation.
        # The second one is the type wise reducer, could be "sum", "max",
        # "min", "mean", "stack"
        G.multi_update_all(funcs, 'sum')
        # return the updated node feature dictionary
        return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}

###############################################################################
# Create a simple GNN by stacking two ``HeteroRGCNLayer``. Since the
# nodes do not have input features, make their embeddings trainable.

class HeteroRGCN(nn.Module):
    def __init__(self, G, in_size, hidden_size, out_size):
        super(HeteroRGCN, self).__init__()
        # Use trainable node embeddings as featureless inputs.
        embed_dict = {ntype : nn.Parameter(torch.Tensor(G.number_of_nodes(ntype), in_size))
                      for ntype in G.ntypes}
        for key, embed in embed_dict.items():
            nn.init.xavier_uniform_(embed)
        self.embed = nn.ParameterDict(embed_dict)
        # create layers
        self.layer1 = HeteroRGCNLayer(in_size, hidden_size, G.etypes)
        self.layer2 = HeteroRGCNLayer(hidden_size, out_size, G.etypes)

    def forward(self, G):
        h_dict = self.layer1(G, self.embed)
        h_dict = {k : F.leaky_relu(h) for k, h in h_dict.items()}
        h_dict = self.layer2(G, h_dict)
        # get paper logits
        return h_dict['paper']

###############################################################################
# Train and evaluate
# ~~~~~~~~~~~~~~~~~~
# Train and evaluate this network.

# Create the model. The output has three logits for three classes.
model = HeteroRGCN(G, 10, 10, 3)

opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

best_val_acc = 0
best_test_acc = 0

for epoch in range(100):
    logits = model(G)
    # The loss is computed only for labeled nodes.
    loss = F.cross_entropy(logits[train_idx], labels[train_idx])

    pred = logits.argmax(1)
    train_acc = (pred[train_idx] == labels[train_idx]).float().mean()
    val_acc = (pred[val_idx] == labels[val_idx]).float().mean()
    test_acc = (pred[test_idx] == labels[test_idx]).float().mean()

    if best_val_acc < val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc

    opt.zero_grad()
    loss.backward()
    opt.step()

    if epoch % 5 == 0:
        print('Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), Test Acc %.4f (Best %.4f)' % (
            loss.item(),
            train_acc.item(),
            val_acc.item(),
            best_val_acc.item(),
            test_acc.item(),
            best_test_acc.item(),
        ))

###############################################################################
# What's next?
# ------------
# * Check out our full implementation in PyTorch
#   `here <https://github.com/dmlc/dgl/tree/master/examples/pytorch/rgcn-hetero>`_.
#
# * We also provide the following model examples:
#
#   * `Graph Convolutional Matrix Completion <https://arxiv.org/abs/1706.02263>_`,
#     which we implement in MXNet
#     `here <https://github.com/dmlc/dgl/tree/v0.4.0/examples/mxnet/gcmc>`_.
#
#   * `Heterogeneous Graph Attention Network <https://arxiv.org/abs/1903.07293>`_
#     requires transforming a heterograph into a homogeneous graph according to
#     a given metapath (i.e. a path template consisting of edge types).  We
#     provide :func:`dgl.transform.metapath_reachable_graph` to do this.  See full
#     implementation
#     `here <https://github.com/dmlc/dgl/tree/master/examples/pytorch/han>`_.
#
#   * `Metapath2vec <https://dl.acm.org/citation.cfm?id=3098036>`_ requires
#     generating random walk paths according to a given metapath.  Please
#     refer to the full metapath2vec implementation
#     `here <https://github.com/dmlc/dgl/tree/master/examples/pytorch/metapath2vec>`_.
#
# * :doc:`Full heterograph API reference <../../api/python/heterograph>`.
