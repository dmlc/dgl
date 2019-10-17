"""
.. _model-sampling:

NodeFlow and Sampling
=======================================

**Author**: Ziyue Huang, Da Zheng, Quan Gan, Jinjing Zhou, Zheng Zhang
"""
################################################################################################
#
# GCN
# ~~~
#
# In an :math:`L`-layer graph convolution network (GCN), given a graph
# :math:`G=(V, E)`, represented as an adjacency matrix :math:`A`, with
# node features :math:`H^{(0)} = X \in \mathbb{R}^{|V| \times d}`, the
# hidden feature of a node :math:`v` in :math:`(l+1)`-th layer
# :math:`h_v^{(l+1)}` depends on the features of all its neighbors in the
# previous layer :math:`h_u^{(l)}`:
#
# .. math::
#
#
#    z_v^{(l+1)} = \sum_{u \in \mathcal{N}(v)} \tilde{A}_{uv} h_u^{(l)} \qquad h_v^{(l+1)} = \sigma ( z_v^{(l+1)} W^{(l)})
#
# where :math:`\mathcal{N}(v)` is the neighborhood of :math:`v`,
# :math:`\tilde{A}` could be any normalized version of :math:`A` such as
# :math:`D^{-1} A` in Kipf et al., :math:`\sigma(\cdot)` is an activation
# function, and :math:`W^{(l)}` is a trainable parameter of the
# :math:`l`-th layer.
#
# In the node classification task we minimize the following loss:
#
# .. math::
#
#
#    \frac{1}{\vert \mathcal{V}_\mathcal{L} \vert} \sum_{v \in \mathcal{V}_\mathcal{L}} f(y_v, z_v^{(L)})
#
# where :math:`y_v` is the label of :math:`v`, and :math:`f(\cdot, \cdot)`
# is a loss function, e.g., cross entropy loss.
#
# While training GCN on the full graph, each node aggregates the hidden
# features of its neighbors to compute its hidden feature in the next
# layer.
#
# In this tutorial, we will run GCN on the Reddit dataset constructed by `Hamilton et
# al. <https://arxiv.org/abs/1706.02216>`__, wherein the nodes are posts
# and edges are established if two nodes are commented by a same user. The
# task is to predict the category that a post belongs to. This graph has
# 233K nodes, 114.6M edges and 41 categories. Let's first load the Reddit graph.
#
import numpy as np
import dgl
import dgl.function as fn
from dgl import DGLGraph
from dgl.data import RedditDataset
import mxnet as mx
from mxnet import gluon

# Load MXNet as backend
dgl.load_backend('mxnet')

# load dataset
data = RedditDataset(self_loop=True)
train_nid = mx.nd.array(np.nonzero(data.train_mask)[0]).astype(np.int64)
features = mx.nd.array(data.features)
in_feats = features.shape[1]
labels = mx.nd.array(data.labels)
n_classes = data.num_labels

# construct DGLGraph and prepare related data
g = DGLGraph(data.graph, readonly=True)
g.ndata['features'] = features

################################################################################################
# Here we define the node UDF which has a fully-connected layer:
#

class NodeUpdate(gluon.Block):
    def __init__(self, in_feats, out_feats, activation=None):
        super(NodeUpdate, self).__init__()
        self.dense = gluon.nn.Dense(out_feats, in_units=in_feats)
        self.activation = activation

    def forward(self, node):
        h = node.data['h']
        h = self.dense(h)
        if self.activation:
            h = self.activation(h)
        return {'activation': h}

################################################################################################
# In DGL, we implement GCN on the full graph with ``update_all`` in ``DGLGraph``.
# The following code performs two-layer GCN on the Reddit graph.
#

# number of GCN layers
L = 2
# number of hidden units of a fully connected layer
n_hidden = 64

layers = [NodeUpdate(g.ndata['features'].shape[1], n_hidden, mx.nd.relu),
          NodeUpdate(n_hidden, n_hidden, mx.nd.relu)]
for layer in layers:
    layer.initialize()

h = g.ndata['features']
for i in range(L):
    g.ndata['h'] = h
    g.update_all(message_func=fn.copy_src(src='h', out='m'),
                 reduce_func=fn.sum(msg='m', out='h'),
                 apply_node_func=lambda node: {'h': layers[i](node)['activation']})
    h = g.ndata.pop('h')

##############################################################################
# NodeFlow
# ~~~~~~~~~~~~~~~~~
#
# As the graph scales up to billions of nodes or edges, training on the
# full graph would no longer be efficient or even feasible.
#
# Mini-batch training allows us to control the computation and memory
# usage within some budget. The training loss for each iteration is
#
# .. math::
#
#    \frac{1}{\vert \tilde{\mathcal{V}}_\mathcal{L} \vert} \sum_{v \in \tilde{\mathcal{V}}_\mathcal{L}} f(y_v, z_v^{(L)})
#
# where :math:`\tilde{\mathcal{V}}_\mathcal{L}` is a subset sampled from
# the total labeled nodes :math:`\mathcal{V}_\mathcal{L}` uniformly at
# random.
#
# Stemming from the labeled nodes :math:`\tilde{\mathcal{V}}_\mathcal{L}`
# in a mini-batch and tracing back to the input forms a computational
# dependency graph (a directed acyclic graph or DAG in short), which
# captures the computation flow of :math:`Z^{(L)}`.
#
# In the example below, a mini-batch to compute the hidden features of
# node D in layer 2 requires hidden features of A, B, E, G in layer 1,
# which in turn requires hidden features of C, D, F in layer 0.
#
# |image0|
#
# For that purpose, we define ``NodeFlow`` to represent this computation
# flow.
#
# ``NodeFlow`` is a type of layered graph, where nodes are organized in
# :math:`L + 1` sequential *layers*, and edges only exist between adjacent
# layers, forming *blocks*. We construct ``NodeFlow`` backwards, starting
# from the last layer with all the nodes whose hidden features are
# requested. The set of nodes the next layer depends on forms the previous
# layer. An edge connects a node in the previous layer to another in the
# next layer iff the latter depends on the former. We repeat such process
# until all :math:`L + 1` layers are constructed. The feature of nodes in
# each layer, and that of edges in each block, are stored as separate
# tensors.
#
# .. raw:: html
#
# ``NodeFlow`` provides ``block_compute`` for per-block computation, which
# triggers computation and data propogation from the lower layer to the
# next upper layer.
#

##############################################################################
# Neighbor Sampling
# ~~~~~~~~~~~~~~~~~
#
# Real-world graphs often have nodes with large degree, meaning that a
# moderately deep (e.g. 3 layers) GCN would often depend on input features
# of the entire graph, even if the computation only depends on outputs of
# a few nodes, hence its cost-ineffectiveness.
#
# Sampling methods mitigate this computational problem by reducing the
# receptive field effectively. Fig-c above shows one such example.
#
# Instead of using all the :math:`L`-hop neighbors of a node :math:`v`,
# `Hamilton et al. <https://arxiv.org/abs/1706.02216>`__ propose *neighbor
# sampling*, which randomly samples a few neighbors
# :math:`\hat{\mathcal{N}}^{(l)}(v)` to estimate the aggregation
# :math:`z_v^{(l+1)}` of its total neighbors :math:`\mathcal{N}(v)` in
# :math:`l`-th GCN layer, by an unbiased estimator
# :math:`\hat{z}_v^{(l+1)}`
#
# .. math::
#
#
#    \hat{z}_v^{(l+1)} = \frac{\vert \mathcal{N}(v) \vert }{\vert \hat{\mathcal{N}}^{(l)}(v) \vert} \sum_{u \in \hat{\mathcal{N}}^{(l)}(v)} \tilde{A}_{uv} \hat{h}_u^{(l)} \qquad
#    \hat{h}_v^{(l+1)} = \sigma ( \hat{z}_v^{(l+1)} W^{(l)} )
#
# Let :math:`D^{(l)}` be the number of neighbors to be sampled for each
# node at the :math:`l`-th layer, then the receptive field size of each
# node can be controlled under :math:`\prod_{i=0}^{L-1} D^{(l)}` by
# *neighbor sampling*.
#

##############################################################################
# We then implement *neighbor smapling* by ``NodeFlow``:
#

class GCNSampling(gluon.Block):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCNSampling, self).__init__(**kwargs)
        self.dropout = dropout
        self.n_layers = n_layers
        with self.name_scope():
            self.layers = gluon.nn.Sequential()
            # input layer
            self.layers.add(NodeUpdate(in_feats, n_hidden, activation))
            # hidden layers
            for i in range(1, n_layers-1):
                self.layers.add(NodeUpdate(n_hidden, n_hidden, activation))
            # output layer
            self.layers.add(NodeUpdate(n_hidden, n_classes))

    def forward(self, nf):
        nf.layers[0].data['activation'] = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = nf.layers[i].data.pop('activation')
            if self.dropout:
                h = mx.nd.Dropout(h, p=self.dropout)
            nf.layers[i].data['h'] = h
            # block_compute() computes the feature of layer i given layer
            # i-1, with the given message, reduce, and apply functions.
            # Here, we essentially aggregate the neighbor node features in
            # the previous layer, and update it with the `layer` function.
            nf.block_compute(i,
                             fn.copy_src(src='h', out='m'),
                             lambda node : {'h': node.mailbox['m'].mean(axis=1)},
                             layer)
        h = nf.layers[-1].data.pop('activation')
        return h

##############################################################################
# DGL provides ``NeighborSampler`` to construct the ``NodeFlow`` for a
# mini-batch according to the computation logic of neighbor sampling.
# ``NeighborSampler``
# returns an iterator that generates a ``NodeFlow`` each time. This function
# has many options to give users opportunities to customize the behavior
# of the neighbor sampler, including the number of neighbors to sample,
# the number of hops to sample, etc. Please see `its API
# document <https://doc.dgl.ai/api/python/sampler.html>`__ for more
# details.
#

# dropout probability
dropout = 0.2
# batch size
batch_size = 1000
# number of neighbors to sample
num_neighbors = 4
# number of epochs
num_epochs = 1

# initialize the model and cross entropy loss
model = GCNSampling(in_feats, n_hidden, n_classes, L,
                    mx.nd.relu, dropout, prefix='GCN')
model.initialize()
loss_fcn = gluon.loss.SoftmaxCELoss()

# use adam optimizer
trainer = gluon.Trainer(model.collect_params(), 'adam',
                        {'learning_rate': 0.03, 'wd': 0})

for epoch in range(num_epochs):
    i = 0
    for nf in dgl.contrib.sampling.NeighborSampler(g, batch_size,
                                                   num_neighbors,
                                                   neighbor_type='in',
                                                   shuffle=True,
                                                   num_hops=L,
                                                   seed_nodes=train_nid):
        # When `NodeFlow` is generated from `NeighborSampler`, it only contains
        # the topology structure, on which there is no data attached.
        # Users need to call `copy_from_parent` to copy specific data,
        # such as input node features, from the original graph.
        nf.copy_from_parent()
        with mx.autograd.record():
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).astype('int64')
            batch_labels = labels[batch_nids]
            # cross entropy loss
            loss = loss_fcn(pred, batch_labels)
            loss = loss.sum() / len(batch_nids)
        # backward
        loss.backward()
        # optimization
        trainer.step(batch_size=1)
        print("Epoch[{}]: loss {}".format(epoch, loss.asscalar()))
        i += 1
        # We only train the model with 32 mini-batches just for demonstration.
        if i >= 32:
            break

##############################################################################
# Control Variate
# ~~~~~~~~~~~~~~~
#
# The unbiased estimator :math:`\hat{Z}^{(\cdot)}` used in *neighbor
# sampling* might suffer from high variance, so it still requires a
# relatively large number of neighbors, e.g. \ :math:`D^{(0)}=25` and
# :math:`D^{(1)}=10` in `Hamilton et
# al. <https://arxiv.org/abs/1706.02216>`__. With *control variate*, a
# standard variance reduction technique widely used in Monte Carlo
# methods, 2 neighbors for a node seems sufficient.
#
# *Control variate* method works as follows: given a random variable
# :math:`X` and we wish to estimate its expectation
# :math:`\mathbb{E} [X] = \theta`, it finds another random variable
# :math:`Y` which is highly correlated with :math:`X` and whose
# expectation :math:`\mathbb{E} [Y]` can be easily computed. The *control
# variate* estimator :math:`\tilde{X}` is
#
# .. math::
#
#    \tilde{X} = X - Y + \mathbb{E} [Y] \qquad \mathbb{VAR} [\tilde{X}] = \mathbb{VAR} [X] + \mathbb{VAR} [Y] - 2 \cdot \mathbb{COV} [X, Y]
#
# If :math:`\mathbb{VAR} [Y] - 2\mathbb{COV} [X, Y] < 0`, then
# :math:`\mathbb{VAR} [\tilde{X}] < \mathbb{VAR} [X]`.
#
# `Chen et al. <https://arxiv.org/abs/1710.10568>`__ proposed a *control
# variate* based estimator used in GCN training, by using history
# :math:`\bar{H}^{(l)}` of the nodes which are not sampled, the modified
# estimator :math:`\hat{z}_v^{(l+1)}` is
#
# .. math::
#
#
#    \hat{z}_v^{(l+1)} = \frac{\vert \mathcal{N}(v) \vert }{\vert \hat{\mathcal{N}}^{(l)}(v) \vert} \sum_{u \in \hat{\mathcal{N}}^{(l)}(v)} \tilde{A}_{uv} ( \hat{h}_u^{(l)} - \bar{h}_u^{(l)} ) + \sum_{u \in \mathcal{N}(v)} \tilde{A}_{uv} \bar{h}_u^{(l)} \\
#    \hat{h}_v^{(l+1)} = \sigma ( \hat{z}_v^{(l+1)} W^{(l)} )
#
# This method can also be *conceptually* implemented in DGL as shown
# below,
#

have_large_memory = False
# The control-variate sampling code below needs to run on a large-memory
# machine for the Reddit graph.
if have_large_memory:
    g.ndata['h_0'] = features
    for i in range(L):
        g.ndata['h_{}'.format(i+1)] = mx.nd.zeros((features.shape[0], n_hidden))
    # With control-variate sampling, we only need to sample 2 neighbors to train GCN.
    for nf in dgl.contrib.sampling.NeighborSampler(g, batch_size, expand_factor=2,
                                                   neighbor_type='in', num_hops=L,
                                                   seed_nodes=train_nid):
        for i in range(nf.num_blocks):
            # aggregate history on the original graph
            g.pull(nf.layer_parent_nid(i+1),
                   fn.copy_src(src='h_{}'.format(i), out='m'),
                   lambda node: {'agg_h_{}'.format(i): node.mailbox['m'].mean(axis=1)})
        nf.copy_from_parent()
        h = nf.layers[0].data['features']
        for i in range(nf.num_blocks):
            prev_h = nf.layers[i].data['h_{}'.format(i)]
            # compute delta_h, the difference of the current activation and the history
            nf.layers[i].data['delta_h'] = h - prev_h
            # refresh the old history
            nf.layers[i].data['h_{}'.format(i)] = h.detach()
            # aggregate the delta_h
            nf.block_compute(i,
                             fn.copy_src(src='delta_h', out='m'),
                             lambda node: {'delta_h': node.data['m'].mean(axis=1)})
            delta_h = nf.layers[i + 1].data['delta_h']
            agg_h = nf.layers[i + 1].data['agg_h_{}'.format(i)]
            # control variate estimator
            nf.layers[i + 1].data['h'] = delta_h + agg_h
            nf.apply_layer(i + 1, lambda node : {'h' : layer(node.data['h'])})
            h = nf.layers[i + 1].data['h']
        # update history
        nf.copy_to_parent()

##############################################################################
# You can see full example here, `MXNet
# code <https://github.com/dmlc/dgl/blob/master/examples/mxnet/sampling/>`__
# and `PyTorch
# code <https://github.com/dmlc/dgl/tree/master/examples/pytorch/sampling>`__.
#
# Below shows the performance of graph convolution network and GraphSage
# with neighbor sampling and control variate sampling on the Reddit
# dataset. Our GraphSage with control variate sampling, when sampling one
# neighbor, can achieve over 96% test accuracy. |image1|
#
# More APIs
# ~~~~~~~~~
#
# In fact, ``block_compute`` is one of the APIs that comes with
# ``NodeFlow``, which provides flexibility to research new ideas. The
# computation flow underlying a DAG can be executed in one sweep, by
# calling ``prop_flows``.
#
# ``prop_flows`` accepts a list of UDFs. The code below defines node update UDFs
# for each layer and computes a simplified version of GCN with neighbor sampling.
#

apply_node_funcs = [
    lambda node : {'h' : layers[0](node)['activation']},
    lambda node : {'h' : layers[1](node)['activation']},
]
for nf in dgl.contrib.sampling.NeighborSampler(g, batch_size, num_neighbors,
                                               neighbor_type='in', num_hops=L,
                                               seed_nodes=train_nid):
    nf.copy_from_parent()
    nf.layers[0].data['h'] = nf.layers[0].data['features']
    nf.prop_flow(fn.copy_src(src='h', out='m'),
                 fn.sum(msg='m', out='h'), apply_node_funcs)

##############################################################################
# Internally, ``prop_flow`` triggers the computation by fusing together
# all the block computations, from the input to the top. The main
# advantages of this API are 1) simplicity, 2) allowing more system-level
# optimization in the future.
#
# .. |image0| image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/sampling/NodeFlow.png
# .. |image1| image:: https://s3.us-east-2.amazonaws.com/dgl.ai/tutorial/sampling/sampling_result.png
#
