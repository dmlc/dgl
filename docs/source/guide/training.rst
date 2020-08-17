.. _guide-training:

Training Graph Neural Networks
==============================

Overview
--------

This chapter discusses how to train a graph neural network for node
classification, edge classification, link prediction, and graph
classification for small graph(s), by message passing methods introduced
in :ref:`guide-message-passing` and neural network modules introduced in
:ref:`guide-nn`.

This chapter assumes that your graph as well as all of its node and edge
features can fit into GPU; see :ref:`guide-minibatch` if they cannot.

The following text assumes that the graph(s) and node/edge features are
already prepared. If you plan to use the dataset DGL provides or other
compatible ``DGLDataset`` as is described in :ref:`guide-data-pipeline`, you can
get the graph for a single-graph dataset with something like

.. code:: python

    import dgl
    
    dataset = dgl.data.CiteseerGraphDataset()
    graph = dataset[0]


Note: In this chapter we will use PyTorch as backend.

Heterogeneous Graphs
~~~~~~~~~~~~~~~~~~~~

Sometimes you would like to work on heterogeneous graphs. Here we take a
synthetic heterogeneous graph as an example for demonstrating node
classification, edge classification, and link prediction tasks.

The synthetic heterogeneous graph ``hetero_graph`` has these edge types:

-  ``('user', 'follow', 'user')``
-  ``('user', 'followed-by', 'user')``
-  ``('user', 'click', 'item')``
-  ``('item', 'clicked-by', 'user')``
-  ``('user', 'dislike', 'item')``
-  ``('item', 'disliked-by', 'user')``

.. code:: python

    import numpy as np
    import torch
    
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10
    
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)
    
    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})
    
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)

.. _guide-training-node-classification:

Node Classification/Regression
------------------------------

One of the most popular and widely adopted tasks for graph neural
networks is node classification, where each node in the
training/validation/test set is assigned a ground truth category from a
set of predefined categories. Node regression is similar, where each
node in the training/validation/test set is assigned a ground truth
number.

Overview
~~~~~~~~

To classify nodes, graph neural network performs message passing
discussed in :ref:`guide-message-passing` to utilize the node’s own
features, but also its neighboring node and edge features. Message
passing can be repeated multiple rounds to incorporate information from
larger range of neighborhood.

Writing neural network model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL provides a few built-in graph convolution modules that can perform
one round of message passing. In this guide, we choose
:class:`dgl.nn.pytorch.SAGEConv` (also available in MXNet and Tensorflow),
the graph convolution module for GraphSAGE.

Usually for deep learning models on graphs we need a multi-layer graph
neural network, where we do multiple rounds of message passing. This can
be achieved by stacking graph convolution modules as follows.

.. code:: python

    # Contruct a two-layer GNN model
    import dgl.nn as dglnn
    import torch.nn as nn
    import torch.nn.functional as F
    class SAGE(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(
                in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
            self.conv2 = dglnn.SAGEConv(
                in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = F.relu(h)
            h = self.conv2(graph, h)
            return h

Note that you can use the model above for not only node classification,
but also obtaining hidden node representations for other downstream
tasks such as
:ref:`guide-training-edge-classification`,
:ref:`guide-training-link-prediction`, or
:ref:`guide-training-graph-classification`.

For a complete list of built-in graph convolution modules, please refer
to :ref:`apinn`.

For more details in how DGL
neural network modules work and how to write a custom neural network
module with message passing please refer to the example in :ref:`guide-nn`.

Training loop
~~~~~~~~~~~~~

Training on the full graph simply involves a forward propagation of the
model defined above, and computing the loss by comparing the prediction
against ground truth labels on the training nodes.

This section uses a DGL built-in dataset
:class:`dgl.data.CiteseerGraphDataset` to
show a training loop. The node features
and labels are stored on its graph instance, and the
training-validation-test split are also stored on the graph as boolean
masks. This is similar to what you have seen in :ref:`guide-data-pipeline`.

.. code:: python

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

The following is an example of evaluating your model by accuracy.

.. code:: python

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

You can then write our training loop as follows.

.. code:: python

    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        model.train()
        # forward propagation by using all nodes
        logits = model(graph, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels, valid_mask)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in this example.


`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py>`__
provides an end-to-end homogeneous graph node classification example.
You could see the corresponding model implementation is in the
``GraphSAGE`` class in the example with adjustable number of layers,
dropout probabilities, and customizable aggregation functions and
nonlinearities.

.. _guide-training-rgcn-node-classification:

Heterogeneous graph
~~~~~~~~~~~~~~~~~~~

If your graph is heterogeneous, you may want to gather message from
neighbors along all edge types. You can use the module
:class:`dgl.nn.pytorch.HeteroGraphConv` (also available in MXNet and Tensorflow)
to perform message passing
on all edge types, then combining different graph convolution modules
for each edge type.

The following code will define a heterogeneous graph convolution module
that first performs a separate graph convolution on each edge type, then
sums the message aggregations on each edge type as the final result for
all node types.

.. code:: python

    # Define a Heterograph Conv model
    import dgl.nn as dglnn
    
    class RGCN(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h

``dgl.nn.HeteroGraphConv`` takes in a dictionary of node types and node
feature tensors as input, and returns another dictionary of node types
and node features.

So given that we have the user and item features in the example above.

.. code:: python

    model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    labels = hetero_graph.nodes['user'].data['label']
    train_mask = hetero_graph.nodes['user'].data['train_mask']

One can simply perform a forward propagation as follows:

.. code:: python

    node_features = {'user': user_feats, 'item': item_feats}
    h_dict = model(hetero_graph, {'user': user_feats, 'item': item_feats})
    h_user = h_dict['user']
    h_item = h_dict['item']

Training loop is the same as the one for homogeneous graph, except that
now you have a dictionary of node representations from which you compute
the predictions. For instance, if you are only predicting the ``user``
nodes, you can just extract the ``user`` node embeddings from the
returned dictionary:

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(hetero_graph, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # Compute validation accuracy.  Omitted in this example.
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in the example.


DGL provides an end-to-end example of
`RGCN <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py>`__
for node classification. You can see the definition of heterogeneous
graph convolution in ``RelGraphConvLayer`` in the `model implementation
file <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py>`__.

.. _guide-training-edge-classification:

Edge Classification/Regression
------------------------------

Sometimes you wish to predict the attributes on the edges of the graph,
or even whether an edge exists or not between two given nodes. In that
case, you would like to have an *edge classification/regression* model.

Here we generate a random graph for edge prediction as a demonstration.

.. code:: ipython3

    src = np.random.randint(0, 100, 500)
    dst = np.random.randint(0, 100, 500)
    # make it symmetric
    edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # synthetic node and edge features, as well as edge labels
    edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
    edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
    edge_pred_graph.edata['label'] = torch.randn(1000)
    # synthetic train-validation-test splits
    edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

Overview
~~~~~~~~

From the previous section you have learned how to do node classification
with a multilayer GNN. The same technique can be applied for computing a
hidden representation of any node. The prediction on edges can then be
derived from the representation of their incident nodes.

The most common case of computing the prediction on an edge is to
express it as a parameterized function of the representation of its
incident nodes, and optionally the features on the edge itself.

Model Implementation Difference from Node Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming that you compute the node representation with the model from
the previous section, you only need to write another component that
computes the edge prediction with the
:meth:`~dgl.DGLGraph.apply_edges` method.

For instance, if you would like to compute a score for each edge for
edge regression, the following code computes the dot product of incident
node representations on each edge.

.. code:: python

    import dgl.function as fn
    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

One can also write a prediction function that predicts a vector for each
edge with an MLP. Such vector can be used in further downstream tasks,
e.g. as logits of a categorical distribution.

.. code:: python

    class MLPPredictor(nn.Module):
        def __init__(self, in_features, out_classes):
            super().__init__()
            self.W = nn.Linear(in_features * 2, out_classes)
    
        def apply_edges(self, edges):
            h_u = edges.src['h']
            h_v = edges.dst['h']
            score = self.W(torch.cat([h_u, h_v], 1))
            return {'score': score}
    
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

Training loop
~~~~~~~~~~~~~

Given the node representation computation model and an edge predictor
model, we can easily write a full-graph training loop where we compute
the prediction on all edges.

The following example takes ``SAGE`` in the previous section as the node
representation computation model and ``DotPredictor`` as an edge
predictor model.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, x):
            h = self.sage(g, x)
            return self.pred(g, h)

In this example, we also assume that the training/validation/test edge
sets are identified by boolean masks on edges. This example also does
not include early stopping and model saving.

.. code:: python

    node_features = edge_pred_graph.ndata['feature']
    edge_label = edge_pred_graph.edata['label']
    train_mask = edge_pred_graph.edata['train_mask']
    model = Model(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(edge_pred_graph, node_features)
        loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


Heterogeneous graph
~~~~~~~~~~~~~~~~~~~

Edge classification on heterogeneous graphs is not very different from
that on homogeneous graphs. If you wish to perform edge classification
on one edge type, you only need to compute the node representation for
all node types, and predict on that edge type with
:meth:`~dgl.DGLGraph.apply_edges` method.

For example, to make ``DotProductPredictor`` work on one edge type of a
heterogeneous graph, you only need to specify the edge type in
``apply_edges`` method.

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each edge type computed from
            # the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

You can similarly write a ``HeteroMLPPredictor``.

.. code:: python

    class MLPPredictor(nn.Module):
        def __init__(self, in_features, out_classes):
            super().__init__()
            self.W = nn.Linear(in_features * 2, out_classes)
    
        def apply_edges(self, edges):
            h_u = edges.src['h']
            h_v = edges.dst['h']
            score = self.W(torch.cat([h_u, h_v], 1))
            return {'score': score}
    
        def forward(self, graph, h, etype):
            # h contains the node representations computed from the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(self.apply_edges, etype=etype)
                return graph.edges[etype].data['score']

The end-to-end model that predicts a score for each edge on a single
edge type will look like this:

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype)

Using the model simply involves feeding the model a dictionary of node
types and features.

.. code:: python

    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    label = hetero_graph.edges['click'].data['label']
    train_mask = hetero_graph.edges['click'].data['train_mask']
    node_features = {'user': user_feats, 'item': item_feats}

Then the training loop looks almost the same as that in homogeneous
graph. For instance, if you wish to predict the edge labels on edge type
``click``, then you can simply do

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(hetero_graph, node_features, 'click')
        loss = ((pred[train_mask] - label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


Predicting Edge Type of an Existing Edge on a Heterogeneous Graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may want to predict which type an existing edge belongs
to.

For instance, given the heterogeneous graph above, your task is given an
edge connecting a user and an item, predict whether the user would
``click`` or ``dislike`` an item.

This is a simplified version of rating prediction, which is common in
recommendation literature.

You can use a heterogeneous graph convolution network to obtain the node
representations. For instance, you can still use the RGCN above for this
purpose.

To predict the type of an edge, you can simply repurpose the
``HeteroDotProductPredictor`` above so that it takes in another graph
with only one edge type that “merges” all the edge types to be
predicted, and emits the score of each type for every edge.

In the example here, you will need a graph that has two node types
``user`` and ``item``, and one single edge type that “merges” all the
edge types from ``user`` and ``item``, i.e. ``like`` and ``dislike``.
This can be conveniently created using
:meth:`relation slicing <dgl.DGLGraph.__getitem__>`.

.. code:: python

    dec_graph = hetero_graph['user', :, 'item']

Since the statement above also returns the original edge types as a
feature named ``dgl.ETYPE``, we can use that as labels.

.. code:: python

    edge_label = dec_graph.edata[dgl.ETYPE]

Given the graph above as input to the edge type predictor module, you
can write your predictor module as follows.

.. code:: python

    class HeteroMLPPredictor(nn.Module):
        def __init__(self, in_dims, n_classes):
            super().__init__()
            self.W = nn.Linear(in_dims * 2, n_classes)
    
        def apply_edges(self, edges):
            x = torch.cat([edges.src['h'], edges.dst['h']], 1)
            y = self.W(x)
            return {'score': y}
    
        def forward(self, graph, h):
            # h contains the node representations for each edge type computed from
            # the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

The model that combines the node representation module and the edge type
predictor module is the following:

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroMLPPredictor(out_features, len(rel_names))
        def forward(self, g, x, dec_graph):
            h = self.sage(g, x)
            return self.pred(dec_graph, h)

The training loop then simply be the following:

.. code:: python

    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        logits = model(hetero_graph, node_features, dec_graph)
        loss = F.cross_entropy(logits, edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


DGL provides `Graph Convolutional Matrix
Completion <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
as an example of rating prediction, which is formulated by predicting
the type of an existing edge on a heterogeneous graph. The node
representation module in the `model implementation
file <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
is called ``GCMCLayer``. The edge type predictor module is called
``BiDecoder``. Both of them are more complicated than the setting
described here.

.. _guide-training-link-prediction:

Link Prediction
---------------

In some other settings you may want to predict whether an edge exists
between two given nodes or not. Such model is called a *link prediction*
model.

Overview
~~~~~~~~

A GNN-based link prediction model represents the likelihood of
connectivity between two nodes :math:`u` and :math:`v` as a function of
:math:`\boldsymbol{h}_u^{(L)}` and :math:`\boldsymbol{h}_v^{(L)}`, their
node representation computed from the multi-layer GNN.

.. math::


   y_{u,v} = \phi(\boldsymbol{h}_u^{(L)}, \boldsymbol{h}_v^{(L)})

In this section we refer to :math:`y_{u,v}` the *score* between node
:math:`u` and node :math:`v`.

Training a link prediction model involves comparing the scores between
nodes connected by an edge against the scores between an arbitrary pair
of nodes. For example, given an edge connecting :math:`u` and :math:`v`,
we encourage the score between node :math:`u` and :math:`v` to be higher
than the score between node :math:`u` and a sampled node :math:`v'` from
an arbitrary *noise* distribution :math:`v' \sim P_n(v)`. Such
methodology is called *negative sampling*.

There are lots of loss functions that can achieve the behavior above if
minimized. A non-exhaustive list include:

-  Cross-entropy loss:
   :math:`\mathcal{L} = - \log \sigma (y_{u,v}) - \sum_{v_i \sim P_n(v), i=1,\dots,k}\log \left[ 1 - \sigma (y_{u,v_i})\right]`
-  BPR loss:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} - \log \sigma (y_{u,v} - y_{u,v_i})`
-  Margin loss:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} \max(0, M - y_{u, v} + y_{u, v_i})`,
   where :math:`M` is a constant hyperparameter.

You may find this idea familiar if you know what `implicit
feedback <https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf>`__ or
`noise-contrastive
estimation <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`__
is.

Model Implementation Difference from Edge Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The neural network model to compute the score between :math:`u` and
:math:`v` is identical to the edge regression model described above.

Here is an example of using dot product to compute the scores on edges.

.. code:: python

    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

Training loop
~~~~~~~~~~~~~

Because our score prediction model operates on graphs, we need to
express the negative examples as another graph. The graph will contain
all negative node pairs as edges.

The following shows an example of expressing negative examples as a
graph. Each edge :math:`(u,v)` gets :math:`k` negative examples
:math:`(u,v_i)` where :math:`v_i` is sampled from a uniform
distribution.

.. code:: python

    def construct_negative_graph(graph, k):
        src, dst = graph.edges()
    
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,))
        return dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())

The model that predicts edge scores is the same as that of edge
classification/regression.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, neg_g, x):
            h = self.sage(g, x)
            return self.pred(g, h), self.pred(neg_g, h)

The training loop then repeatedly constructs the negative graph and
computes loss.

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()
    
    node_features = graph.ndata['feat']
    n_features = node_features.shape[1]
    k = 5
    model = Model(n_features, 100, 100)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        negative_graph = construct_negative_graph(graph, k)
        pos_score, neg_score = model(graph, negative_graph, node_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


After training, the node representation can be obtained via

.. code:: python

    node_embeddings = model.sage(graph, node_features)

There are multiple ways of using the node embeddings. Examples include
training downstream classifiers, or doing nearest neighbor search or
maximum inner product search for relevant entity recommendation.

Heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~

Link prediction on heterogeneous graphs is not very different from that
on homogeneous graphs. The following assumes that we are predicting on
one edge type, and it is easy to extend it to multiple edge types.

For example, you can reuse the ``HeteroDotProductPredictor`` above for
computing the scores of the edges of an edge type for link prediction.

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each edge type computed from
            # the GNN above.
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

To perform negative sampling, one can construct a negative graph for the
edge type you are performing link prediction on as well.

.. code:: python

    def construct_negative_graph(graph, k, etype):
        utype, _, vtype = etype
        src, dst = graph.edges(etype=etype)
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
        return dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

The model is a bit different from that in edge classification on
heterogeneous graphs since you need to specify edge type where you
perform link prediction.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, neg_g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype), self.pred(neg_g, h, etype)

The training loop is similar to that of homogeneous graphs.

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()
    
    k = 5
    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'click', 'item'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


.. _guide-training-graph-classification:

Graph Classification
--------------------

Instead of a big single graph, sometimes we might have the data in the
form of multiple graphs, for example a list of different types of
communities of people. By characterizing the friendships among people in
the same community by a graph, we get a list of graphs to classify. In
this scenario, a graph classification model could help identify the type
of the community, i.e. to classify each graph based on the structure and
overall information.

Overview
~~~~~~~~

The major difference between graph classification and node
classification or link prediction is that the prediction result
characterize the property of the entire input graph. We perform the
message passing over nodes/edges just like the previous tasks, but also
try to retrieve a graph-level representation.

The graph classification proceeds as follows:

.. figure:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
   :alt: Graph Classification Process

   Graph Classification Process

From left to right, the common practice is:

-  Prepare graphs in to a batch of graphs
-  Message passing on the batched graphs to update node/edge features
-  Aggregate node/edge features into a graph-level representation
-  Classification head for the task

Batch of Graphs
^^^^^^^^^^^^^^^

Usually a graph classification task trains on a lot of graphs, and it
will be very inefficient if we use only one graph at a time when
training the model. Borrowing the idea of mini-batch training from
common deep learning practice, we can build a batch of multiple graphs
and send them together for one training iteration.

In DGL, we can build a single batched graph of a list of graphs. This
batched graph can be simply used as a single large graph, with separated
components representing the corresponding original small graphs.

.. figure:: https://data.dgl.ai/tutorial/batch/batch.png
   :alt: Batched Graph

   Batched Graph

Graph Readout
^^^^^^^^^^^^^

Every graph in the data may have its unique structure, as well as its
node and edge features. In order to make a single prediction, we usually
aggregate and summarize over the possibly abundant information. This
type of operation is named *Readout*. Common aggregations include
summation, average, maximum or minimum over all node or edge features.

Given a graph :math:`g`, we can define the average readout aggregation
as

.. math:: h_g = \frac{1}{|\mathcal{V}|}\sum_{v\in \mathcal{V}}h_v

In DGL the corresponding function call is :func:`dgl.readout_nodes`.

Once :math:`h_g` is available, we can pass it through an MLP layer for
classification output.

Writing neural network model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input to the model is the batched graph with node and edge features.
One thing to note is the node and edge features in the batched graph
have no batch dimension. A little special care should be put in the
model:

Computation on a batched graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Next, we discuss the computational properties of a batched graph.

First, different graphs in a batch are entirely separated, i.e. no edge
connecting two graphs. With this nice property, all message passing
functions still have the same results.

Second, the readout function on a batched graph will be conducted over
each graph separately. Assume the batch size is :math:`B` and the
feature to be aggregated has dimension :math:`D`, the shape of the
readout result will be :math:`(B, D)`.

.. code:: python

    g1 = dgl.graph(([0, 1], [1, 0]))
    g1.ndata['h'] = torch.tensor([1., 2.])
    g2 = dgl.graph(([0, 1], [1, 2]))
    g2.ndata['h'] = torch.tensor([1., 2., 3.])
    
    dgl.readout_nodes(g1, 'h')
    # tensor([3.])  # 1 + 2
    
    bg = dgl.batch([g1, g2])
    dgl.readout_nodes(bg, 'h')
    # tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

Finally, each node/edge feature tensor on a batched graph is in the
format of concatenating the corresponding feature tensor from all
graphs.

.. code:: python

    bg.ndata['h']
    # tensor([1., 2., 1., 2., 3.])

Model definition
^^^^^^^^^^^^^^^^

Being aware of the above computation rules, we can define a very simple
model.

.. code:: python

    class Classifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes):
            super(Classifier, self).__init__()
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
            self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g, feat):
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, 'h')
                return self.classify(hg)

Training loop
~~~~~~~~~~~~~

Data Loading
^^^^^^^^^^^^

Once the model’s defined, we can start training. Since graph
classification deals with lots of relative small graphs instead of a big
single one, we usually can train efficiently on stochastic mini-batches
of graphs, without the need to design sophisticated graph sampling
algorithms.

Assuming that we have a graph classification dataset as introduced in
:ref:`guide-data-pipeline`.

.. code:: python

    import dgl.data
    dataset = dgl.data.GINDataset('MUTAG', False)

Each item in the graph classification dataset is a pair of a graph and
its label. We can speed up the data loading process by taking advantage
of the DataLoader, by customizing the collate function to batch the
graphs:

.. code:: python

    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

Then one can create a DataLoader that iterates over the dataset of
graphs in minibatches.

.. code:: python

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

Loop
^^^^

Training loop then simply involves iterating over the dataloader and
updating the model.

.. code:: python

    model = Classifier(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['feats']
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

DGL implements
`GIN <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__
as an example of graph classification. The training loop is inside the
function ``train`` in
```main.py`` <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/main.py>`__.
The model implementation is inside
```gin.py`` <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/gin.py>`__
with more components such as using
:class:`dgl.nn.pytorch.GINConv` (also available in MXNet and Tensorflow)
as the graph convolution layer, batch normalization, etc.

Heterogeneous graph
~~~~~~~~~~~~~~~~~~~

Graph classification with heterogeneous graphs is a little different
from that with homogeneous graphs. Except that you need heterogeneous
graph convolution modules, yoyu also need to aggregate over the nodes of
different types in the readout function.

The following shows an example of summing up the average of node
representations for each node type.

.. code:: python

    class RGCN(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h
    
    class HeteroClassifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g):
            h = g.ndata['feat']
            # Apply graph convolution and activation.
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
    
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                hg = 0
                for ntype in g.ntypes:
                    hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                return self.classify(hg)

The rest of the code is not different from that for homogeneous graphs.

.. code:: python

    model = HeteroClassifier(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            logits = model(batched_graph)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
