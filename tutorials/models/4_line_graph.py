"""
.. _model-line-graph:
Line Graph Neural Network
=====================================

**Author**: `Qi Huang <https://github.com/HQ01>`_, Yu Gai, Zheng Zhang
"""

###########################################################################################
# 
# In :doc:`GCN <1_gcn>` , we demonstrate how to classify nodes on an input
# graph in a semi-supervised setting, using graph convolutional neural network
# as embedding mechanism for graph features.
# In this tutorial, we shift our focus to community detection problem. The
# task of community detection, i.e. graph clustering, consists of partitioning
# the vertices in a graph into clusters in which nodes are more "similar" to
# one another.
#
# To generalize GNN to supervised community detection, Chen et al. introduced
# a line-graph based variation of graph neural network in 
# `Supervised Community Detection with Line Graph Neural Networks <https://arxiv.org/abs/1705.08415>`__. 
# One of the highlight of their model is
# to augment the vanilla graph neural network(GNN) architecture to operate on
# the line graph of edge adajcencies, defined with non-backtracking operator.
#
# In addition to its high performance, LGNN offers an opportunity to
# illustrate how DGL can implement an advanced graph algorithm by flexibly
# mixing vanilla tensor operations, sparse-matrix multiplication and message-
# passing APIs.
#
# In the following sections, we will go through community detection, line
# graph, LGNN, and its implementation.
#
# Supervised Community Detection Task on CORA
# --------------------------------------------
# Community Detection
# ~~~~~~~~~~~~~~~~~~~~
# In community detection task, we cluster "similar" nodes instead of
# "labeling" them. The node similarity is typically described as higher inner
# density in each cluster.
#
# What's the difference between community detection and node classification？
# Comparing to node classification, community detection focuses on retrieving
# cluster information in the graph, rather than assigning a specific label to
# a node. For example, as long as a node is clusetered with its community
# members, it doesn't matter whether the node is assigned as "community A",
# or "community B", while assigning all "great movies" to label "bad movies"
# will be a disaster in a movie network classification task.
#
# What's the difference then, between a community detection algorithm and
# other clustering algorithm such as k-means? Community detection algorithm operates on
# graph-structured data. Comparing to k-means, community detection leverages
# graph structure, instead of simply clustering nodes based on their
# features.
#
# CORA
# ~~~~~
# To be consistent with Graph Convolutional Network tutorial, 
# we use `CORA <https://linqs.soe.ucsc.edu/data>`__ 
# to illustrate a simple community detection task. To refresh our memory, 
# CORA is a scientific publication dataset, with 2708 papers belonging to 7 
# different mahcine learning sub-fields. Here, we formulate CORA as a 
# directed graph, with each node being a paper, and each edge being a 
# citation link (A->B means A cites B). Here is a visualization of the whole 
# CORA dataset.
#
# .. figure:: https://i.imgur.com/X404Byc.png
#    :alt:
#
# CORA naturally contains 7 "classes", and statistics below show that each
# "class" does satisfy our assumption of community, i.e. nodes of same class
# class have higher connection probability among them than with nodes of diferent class.
# The following code snippet verifies that there are more intra-class edges
# than inter-class:

import dgl
from dgl.data import citation_graph as citegrh


data = citegrh.load_cora()
num_classes = 7

import dgl.tutorial_utils as utils
utils.check_intra_prob(data.graph, data.labels, num_classes)

###########################################################################################
# Binary Community Subgraph from CORA -- a Toy Dataset
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Without loss of generality, in this tutorial we limit the scope of our
# task to binary community detection.
# 
# .. note::
#
#    To create a toy binary-community dataset from CORA, We first extract
#    all two-class pairs from the original CORA 7 classes. For each pair, we
#    treat each class as one community, and find the largest subgraph that
#    at least contain one cross-community edge as the training example.Here
#    is a simple example of an extracted binary commnity subraph from CORA.
#    Nodes in blue belong to one community, nodes in red belong to another.
#
# Here is an example:
#

from dgl.data import binary_sub_graph as bsg
from dgl import DGLGraph
train_set = bsg.CORABinary(DGLGraph(data.graph), data.features, data.labels, num_classes=7)
num_train = len(train_set)
[g, lg, g_deg, lg_deg, pm_pd, subfeature, label, equi_label] = train_set[1]
utils.graph_viz(label, g.to_networkx())
###########################################################################################
# Interested readers can go to the original paper to see how to generalize
# to multi communities case.
#
# Community Detection in a Supervised Setting
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Community Detection problem could be tackled with both supervised and
# unsupervised approaches. Same as the original paper, we formulate
# Community Detection in a supervised setting as follows:
#
# - Each training example consists of :math:`(G, L)`, where :math:`G` is a
#   directed graph :math:`(V, E)`. For each node :math:`v` in :math:`V`, we
#   assign a ground truth community label :math:`z_v \in \{0,1\}`.
# - The parameterized model :math:`f(G, \theta)` predicts a label set
#   :math:`\tilde{Z} = f(G)` for nodes :math:`V`.
# - For each example :math:`(G,L)`, the model learn to minimize a specially-
#   designed loss function (equivariant loss) :math:`L_{equivariant} =
#   (\tilde{Z}，Z)`
# Interested readers can check note to see a detalied explanation of the
# equivariant loss.
#
# .. note::
#
#    In this supervised setting, the model naturally predicts a "label" for
#    each community. However, community assignment should be equivariant to
#    label permutations. To acheive this, in each forward process, we take
#    the minimum among losses calcuated from all possible permutations of
#    labels.
#
#    Mathematically, this means
#    :math:`L_{equivariant} = \underset{\pi \in S_c} {min}-\log(\hat{\pi}, \pi)`,
#    where :math:`S_c` is the set of all permutations of labels, and
#    :math:`\hat{\pi}` is the set of predicted labels,
#    :math:`- \log(\hat{\pi},\pi)` denotes negative log likelihood.
#
#    For instance, for a toy graph with node :math:`\{1,2,3,4\}` and
#    community assignment :math:`\{A, A, A, B\}`, with each node's label
#    :math:`l \in \{0,1\}`,The group of all possible permutations
#    :math:`S_c = \{\{0,0,0,1\}, \{1,1,1,0\}\}`.
# 
# Line Graph Neural network: key ideas
# ------------------------------------
# An key innovation in this paper is the use of line-graph.
# Unlike models in previous tutorials, message passing happens not only on the original graph,
# e.g. the binary community subgraph from CORA, but also on the line-graph associated with the original graph.
#
# What's a line-graph ?
# ~~~~~~~~~~~~~~~~~~~~~
# In graph theory, line graph is a graph representation that encodes the
# edge adjacency sturcutre in the original graph.
#
# Specifically, a line-graph `lg` turns an edge of the original graph `g`
# into a node.This is illustrated with the graph below (taken from the
# paper)
# 
# .. figure:: https://i.imgur.com/4WO5jEm.png
#    :alt:
#
# Here, :math:`e_{A}:= （i\rightarrow j）` and :math:`e_{B}:= (j\rightarrow k)`
# are two edges in the original graph :math:`G`. In line graph :math:`G_L`,
# they correspond to nodes :math:`v^{l}_{A}, v^{l}_{B}`.
#
# The next natural question is, how to connect nodes in line-graph？ How to
# connect two "edges"? Here, we use the following connection rule:
#
# Two nodes :math:`v^{l}_{A}`, :math:`v^{l}_{B}` in `lg` are connected if
# the corresponding two edges :math:`e_{A}, e_{B}` in `g` share one and only 
# one node:
# :math:`e_{A}`'s destination node is :math:`e_{B}`'s source node
# (:math:`j`).
# 
# .. note::
#
#    Mathematically, this definition corresponds to a notion called non-
#    backtracking operator:
#    :math:`B_{(i \rightarrow j), (\hat{i} \rightarrow \hat{j})}`
#    :math:`= \begin{cases}
#    1 \text{ if } j = \hat{i}, \hat{j} \neq i\\
#    0 \text{ otherwise} \end{cases}`
#    where an edge is formed if :math:`B_{node1, node2} = 1`.
#
#
# One layer in LGNN -- algorithm sturcture
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# LGNN chains up a series of line-graph neural network layers. The graph
# reprentation :math:`x` and its line-graph companion :math:`y` evolve with
# the dataflow as follows,
# 
# .. figure:: https://i.imgur.com/bZGGIGp.png
#    :alt:
#
# At the :math:`k`-th layer, the :math:`i`-th neuron of the :math:`l`-th
# channel updates its embedding :math:`x^{(k+1)}_{i,l}` with:
#
# .. math::
#    \begin{split}
#    x^{(k+1)}_{i,l} ={}&\rho[x^{(k)}_{i}\theta^{(k)}_{1,l}
#    +(Dx^{(k)})_{i}\theta^{(k)}_{2,l} \\
#    &+\sum^{J-1}_{j=0}(A^{2^{j}}x^{k})_{i}\theta^{(k)}_{3+j,l}\\
#    &+[\{\text{Pm},\text{Pd}\}y^{(k)}]_{i}\theta^{(k)}_{3+J,l}] \\
#    &+\text{skip-connection}
#    \qquad i \in V, l = 1,2,3, ... b_{k+1}/2
#    \end{split}
#
# Then, the line-graph representation :math:`y^{(k+1)}_{i,l}` with,
#
# .. math::
#
#    \begin{split}
#    y^{(k+1)}_{i',l^{'}} = {}&\rho[y^{(k)}_{i^{'}}\gamma^{(k)}_{1,l^{'}}+
#    (D_{L(G)}y^{(k)})_{i^{'}}\gamma^{(k)}_{2,l^{'}}\\
#    &+\sum^{J-1}_{j=0}(A_{L(G)}^{2^{j}}y^{k})_{i}\gamma^{(k)}_{3+j,l^{'}}\\
#    &+[\{\text{Pm},\text{Pd}\}^{T}x^{(k+1)}]_{i^{'}}\gamma^{(k)}_{3+J,l^{'}}]\\
#    &+\text{skip-connection}
#    \qquad i^{'} \in V_{l}, l^{'} = 1,2,3, ... b^{'}_{k+1}/2
#    \end{split}
# Where :math:`\text{skip-connection}` refers to performing the same operation without the non-linearity
# :math:`\rho`, and with linear projection :math:`\theta_\{\frac{b_{k+1}}{2} + 1, ..., b_{k+1}-1, b_{k+1}\}`
# and :math:`\gamma_\{\frac{b_{k+1}}{2} + 1, ..., b_{k+1}-1, b_{k+1}\}`.
#
# Implement LGNN in DGL
# ---------------------
# General idea
# ~~~~~~~~~~~~
# The above equations look intimidating. However, we observe the following:
# 
# - The two equations are symmetric and can be implemented as two instances
#   of the same class with different parameters.
#   Mainly, the first equation operates on graph representation :math:`x`,
#   whereas the sedond operates on line-graph
#   representation :math:`y`. Denote this abstraction as :math:`f`, then
#   the first is :math:`f(x,y; \theta_x)`, and the second
#   is :math:`f(y,x, \theta_y)`. That is, they are parameterized to compute
#   representations of the original graph and its
#   companion line graph, respectively.
#
# - Each equation consists of 4 terms: (take the first as example):
#        - :math:`x^{(k)}\theta^{(k)}_{1,l}`, a linear projection of previous layer's output :math:`x^{(k)}`, denote as
#          :math:`\text{prev}(x)`.
#        - :math:`(Dx^{(k)})\theta^{(k)}_{2,l}`, a linear projection of degree operator on :math:`x^{(k)}`, denote as
#          :math:`\text{deg}(x)`.
#        - :math:`\sum^{J-1}_{j=0}(A^{2^{j}}x^{(k)})\theta^{(k)}_{3+j,l}`, a summation of :math:`2^{j}` adjacency operator on
#          :math:`x^{(k)}`, denote as :math:`\text{sum}(x)`
#        - :math:`[\{Pm,Pd\}y^{(k)}]\theta^{(k)}_{3+J,l}`, fusing another graph's embedding information using incidence matrix
#          :math:`\{Pm, Pd\}`, followed with a linear porjection, denote as :math:`\text{fuse}(y)`.
#
#
# - In addition, each of the terms are performed again with different
#   parameters, and without the nonlinearity after the sum.
#   Therefore, :math:`f` could be written as:
# 
#   .. math::
#      \begin{split}
#      f(x^{(k)},y^{(k)}) = {}\rho[&\text{prev}_{1}(x^{(k-1)}) + \text{deg}_{1}(x^{(k-1)}) +\text{sum}_{1}(x^{k-1})
#      +\text{fuse}_{1}(y^{(k)})]\\
#      +&\text{prev}_{1}(x^{(k-1)}) + \text{deg}_{1}(x^{(k-1)}) +\text{sum}_{1}(x^{k-1}) +\text{fuse}_{1}(y^{(k)})
#      \end{split}
#
# - Two equations are chained up in the following order :
# 
#   .. math::
#      \begin{split}
#      x^{(k+1)} = {}& f(x^{(k)}, y^{(k)})\\
#      y^{(k+1)} = {}& f(y^{(k)}, x^{(k+1)})
#      \end{split}
# 
# With these observations, we proceed to implementation.
# The important point is we are to use different strategies for these terms.
# 
# .. note::
#    For a detailed explanation of :math:`\{Pm, Pd\}`, please go to ``Advanced Topic #2``.
#
# Implementing :math:`\text{prev}` and :math:`\text{deg}` as tensor operation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since linear projection and degree operation are both simply matrix
# multiplication, we can write them as PyTorch tensor operation.
#
# In ``__init__``, we define the projection variables:
# 
# ::
# 
#    self.new_linear = lambda: nn.Linear(feats, out_feats)
#    self.linear_prev, self.linear_deg = self.new_linear(), self.new_linear()
# 
#
# In ``forward()``, :math:`\text{prev}` and :math:`\text{deg}` are the same
# as any other PyTorch tensor operations.
# 
# ::
# 
#   prev_proj = self.linear_prev(feat_a)
# 
#   deg_proj = self.linear_deg(deg * feat_a)
# 
# Implementing :math:`\text{sum}` as message passing in DGL
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# As discussed in GCN tutorial, we can formulate one adjacency operator as
# doing one step message passing. As a generalization, :math:`2^j` adjacency
# operations can be formulated as performing :math:`2^j` step of message
# passing. Therefore, the summation is equivalent to summing nodes'
# representation of :math:`2^j, j=0, 1, 2..` step messsage passing, i.e.
# gathering information in :math:`2^{j}` neighbourhood of each node.
#
# In ``__init__``, we define the projection variables used in each
# :math:`2^j` steps of message passing:
# 
# ::
# 
#   self.new_linear_list = lambda: nn.ModuleList([self.new_linear() for i in range(radius)])
# 
#   self.linear_aggregate = self.new_linear_list()
# 
#
#
# In ``__forward__``, we define the ``sum`` operation as ``aggregate()``:
def aggregate(g, z):
            # initializing list to collect message passing result
            z_list = []
            g.ndata['z'] = z
            # pulling message from 1-hop neighbourhood
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
            z_list.append(g.ndata['z'])
            for i in range(self.radius - 1):
                for j in range(2 ** i):
                    #pulling message from 2^j neighborhood
                    g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
                z_list.append(g.ndata['z'])
            return z_list
#########################################################################
def twoj_hop(feat):
    return sum(linear(z) for linear, z in zip(self.linear_aggregate,
                                              aggregate(g,feat)))
#########################################################################
# 
# Then 
# ::
# 
#   sum_a = twoj_hop(feat_a)
# 
# Implementing :math:`\text{fuse}` as sparse matrix multiplication
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# :math:`\{Pm, Pd\}` is a sparse matrix with only two non-zero entries on
# each column. Therefore, we construct it as a sparse matrix in the dataset, # and implement :math:`\text{fuse}` as a sparse matrix multiplication.
#
# in ``__forward__``:
# 
# ::
# 
#   pmpd_exp = lambda feat : th.matmul(pm_pd, feat)
# 
#   fuse = self.linear_fuse(pmpd_exp(feat_b))
#
# Completing :math:`f(x, y)`
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# Finally, we sum up all the terms together, pass it to skip connection and
# batch-norm.
# 
# ::
#
#   result = prev_proj + deg_proj + sum_a + fuse``
# 
# Then pass result to skip connection: 
# 
# ::
# 
#   result = th.cat([result[:, :n], F.relu(result[:, n:])], 1)``
# 
# Then batch norm
# 
# ::
# 
#   result = self.bn(result) #Batch Normalization.``
# 
#
# Below is the complete code for one LGNN layer's abstraction :math:`f(x,y)`
import torch.nn as nn
import dgl.function as fn
import torch as th
import torch.nn.functional as F
class LGNNCore(nn.Module):
    def __init__(self, feats, out_feats, radius, mode='g'):
        super(LGNNCore, self).__init__()
        self.mode = mode
        self.out_feats = out_feats
        self.radius = radius


        self.new_linear = lambda: nn.Linear(feats, out_feats)
        self.new_linear_other = lambda: nn.Linear(out_feats, out_feats)

        self.new_linear_list = lambda: nn.ModuleList([self.new_linear() for i in range(radius)])
        self.linear_prev, self.linear_deg = self.new_linear(), self.new_linear()
        self.linear_aggregate = self.new_linear_list()


        if (mode == 'g'):
            self.linear_fuse = self.new_linear()
        else:
            self.linear_fuse = self.new_linear_other()


        self.bn = nn.BatchNorm1d(out_feats)


    def forward(self, g, feat_a, feat_b, deg, pm_pd):

        def aggregate(g, z):
            z_list = []
            g.ndata['z'] = z
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
            z_list.append(g.ndata['z'])
            for i in range(self.radius - 1):
                for j in range(2 ** i):
                    g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
                z_list.append(g.ndata['z'])
            return z_list

        prev_proj = self.linear_prev(feat_a)
        deg_proj = self.linear_deg(deg * feat_a)

        twoj_hop = lambda feat : sum(linear(z)
                                     for linear, z in zip(self.linear_aggregate, aggregate(g, feat)))

        pmpd_exp = lambda feat : th.matmul(pm_pd, feat)


        sum_a = twoj_hop(feat_a)
        fuse = self.linear_fuse(pmpd_exp(feat_b))


        result = prev_proj + deg_proj + sum_a + fuse

        n = self.out_feats // 2

        result = th.cat([result[:, :n], F.relu(result[:, n:])], 1) #skip connection
        result = self.bn(result)

        return result

##############################################################################################################
# Chain up LGNN abstractions as a LGNN layer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# To implement:
# 
# .. math::
#    \begin{split}
#    x^{(k+1)} = {}& f(x^{(k)}, y^{(k)})\\
#    y^{(k+1)} = {}& f(y^{(k)}, x^{(k+1)})
#    \end{split}
# We chain up 2 ``LGNNCore`` instances with different parameter in the forward pass.
class LGNNLayer(nn.Module):
    def __init__(self, feats, out_feats, radius):
        super(LGNNLayer, self).__init__()

        self.g_layer = LGNNCore(feats, out_feats, radius, mode='g')
        self.lg_layer = LGNNCore(feats, out_feats, radius, mode='lg')

    def forward(self, g, lg, x, lg_x, deg_g, deg_lg, pm_pd):
        x = self.g_layer(g, x, lg_x, deg_g, pm_pd)
        pm_pd_y = th.transpose(pm_pd, 0, 1)
        lg_x = self.lg_layer(lg, lg_x, x, deg_lg, pm_pd_y) # Here we can add pm_pd_y

        return x, lg_x
########################################################################################
# Chain up LGNN layers
# ~~~~~~~~~~~~~~~~~~~~
# The final LGNN class is defined by chaining up arbitrary number of LGNN layers.
from dgl.tutorial_utils import from_npsp
class LGNN(nn.Module):
    def __init__(self, feats, radius, n_classes):
        """
        Parameters
        ----------
        feats : dimension of intermediate layers
        radius : radius of neighborhood message passing
        n_classes : number of predicted communities, i.e. dimension of last layer
        """
        super(LGNN, self).__init__()
        self.linear = nn.Linear(feats[-1], n_classes)

        self.module_list = nn.ModuleList([LGNNLayer(in_feat, out_feat, radius)
                                          for in_feat, out_feat in zip(
                                              feats[:-1], feats[1:])])
    @from_npsp
    def forward(self, g, lg, deg_g, deg_lg, pm_pd):
        x, lg_x = deg_g, deg_lg
        for module in self.module_list:
            x, lg_x = module(g, lg, x, lg_x, deg_g, deg_lg, pm_pd)
        return self.linear(x)
#########################################################################################
# Training and Inference
# -----------------------
# We first load the data
from torch.utils.data import DataLoader
n_batch_size = 2

indices = list(range(num_train))
training_loader = DataLoader(train_set,
                             n_batch_size,
                             collate_fn=train_set.collate_fn,
                             drop_last=True)
##############################################################
# The line graph construction is done with DGL's line graph API. Here is a code snippet:
# 
# ::
# 
#  self._line_graphs = [g.line_graph(backtracking=False) for g in self._subgraphs]``
# 
# 
#
# ``backracking = false`` means we use non-backtracking operator to define the line graph.
# See code link at here (link to data/binary-sub-grph).
# Here we define a main loop to train a 3 layers LGNN for 20 epochs on the toy-datset.
# We first define a ``step()`` function to describe each step of training:
import time
@from_npsp
def step(i, j, g, lg, deg_g, deg_lg, pmpd, feature, label, equi_label, n_batchsize):
    """ One step of training. """
    t0 = time.time()
    z = model(g, lg, deg_g, deg_lg, pmpd)
    time_forward = time.time() - t0

    z_list = [z]
    equi_labels = [label, equi_label]
    loss = sum(min(F.cross_entropy(z, y) for y in equi_labels) for z in z_list) / n_batchsize

    accu = utils.linegraph_accuracy(z_list, equi_labels)

    optimizer.zero_grad()
    t0 = time.time()
    loss.backward()
    time_backward = time.time() - t0
    optimizer.step()

    return loss, accu, time_forward, time_backward
####################################################################################################
# initialize the model
import torch.optim as optim
n_features = 16
n_layers = 3
radius = 3
lr = 1e-2
K = 2 # num_of_classes
inference_idx = 1
feats = [1] + [n_features]*n_layers + [K]
model = LGNN(feats, radius, K)

optimizer = optim.Adam(model.parameters(), lr=lr)
######################################################################################################
# Below is the main training loop
n_iterations = 20 #main loop
n_epochs = 20
vali_label_change = [] # This is probably not the best practice
total_time = 0
for i in range(n_epochs):
    total_loss, total_accu, s_forward, s_backward = 0, 0, 0, 0
    for j, [g, lg, g_deg, lg_deg, pmpd, subfeature, label, equivariant_label] in enumerate(training_loader):
        loss, accu, t_forward, t_backward = step(i, j, g, lg, g_deg,
                                                 lg_deg, pmpd,
                                                 subfeature, label,
                                                 equivariant_label,
                                                 n_batch_size)
        total_loss += loss
        s_forward += t_forward
        s_backward += t_backward
        total_accu += accu
    total_time += (s_forward + s_backward)

    print("average loss for epoch {} is {}, with avg accu {}, forward time {}, backward time {}".format(i, total_loss/len(training_loader), total_accu/len(training_loader), s_forward, s_backward))
    [g, lg, g_deg, lg_deg, pmpd, subfeature, label, equi_label] = train_set[inference_idx]
    z = model(g, lg, g_deg, lg_deg, pmpd)
    vali_label_change.append(th.max(z, 1)[1])
print("total time {} s, average {}".format(total_time, total_time/n_iterations))
vali_label_change.append(equi_label)
####################################################################################################################
# Visualizing training progress
# -----------------------------
# To intuitively understand the training progress of a LGNN,
# we visualize the network's community prediction on one training example,
# together with the ground truth.
print("visualization on one training example...")
import matplotlib.pyplot as plt

[g, lg, g_deg, lg_deg, pmpd, subfeature, label, equi_label] = train_set[inference_idx]
utils.linegraph_inference_viz(g, lg, g_deg, lg_deg, pmpd,
                                  subfeature, model)
plt.show()
#######################################################################################
# Below is ground truth.
utils.graph_viz(label, g.to_networkx())
plt.show()
#########################################
# We then provide an animation to better understand the process. (40 epochs)
import matplotlib.animation as animation #Save in local disk

[g, lg, g_deg, lg_deg, pm_pd, subfeature, label, equi_label] = train_set[inference_idx]
# utils.animate(g, vali_label_change)
#########################################################################################
# .. figure:: https://i.imgur.com/KDUyE1S.gif 
#    :alt:
# Advanced topic #1: batching
# ---------------------------
# LGNN takes a collection of different graphs.
# Thus, it's natural we use batching to explore parallelism.
# Why is it not done?
#
# As it turned out, we moved batching into the dataloader itself.
#
# In the ``collate_fn`` for PyTorch Dataloader, we batch graphs using DGL's batched_graph API.
# Degree matrices are simply numpy arrays, thus we concatenate them.
# To refresh our memory, DGL batches graphs by merging them into a large graph,
# with each smaller graph's adjacency matrix being a block along the diagonal of the large graph's adjacency matrix.
# We concatentate :math`\{Pm,Pd\}` as block diagonal matrix in corespondance to DGL batched graph API.
from dgl.batched_graph import batch
def collate_fn(self, x):
        subgraph, line_graph, deg_g, deg_lg, pmpd, subfeature, sublabel, equi_label = zip(*x)
        subgraph_batch = batch(subgraph)
        line_graph_batch = batch(line_graph)
        deg_g_batch = np.concatenate(deg_g, axis=0)
        deg_lg_batch = np.concatenate(deg_lg, axis=0)

        self.total = 0

        subfeature_batch = np.concatenate(subfeature, axis=0)
        sublabel_batch = np.concatenate(sublabel, axis=0)
        equilabel_batch = np.concatenate(equi_label, axis=0)

        pmpd_batch = sp.sparse.block_diag(pmpd)

        return subgraph_batch, line_graph_batch, deg_g_batch, deg_lg_batch, pmpd_batch, subfeature_batch, sublabel_batch, equilabel_batch
##########################################################################################################################
# You can check out the complete code here (link to dataloader).
# 
# 
# Advanced topic #2: what's the business with :math:`\{Pm, Pd\}`?
# ----------------------------------------------------------------
# Rougly speaking, there is a relationship between how :math:`g` and
# :math:`lg` (the line graph) working together with loopy brief propagation.
# Here, we implement :math:`\{Pm, Pd\}` as scipy coo sparse matrix in the datset,
# and stack them as tensors when batching. Another batching solution is to
# treat :math:`\{Pm, Pd\}` as the adjacency matrix of a bipartie graph, which maps
# line graph's feature to graph's, and vice versa.


