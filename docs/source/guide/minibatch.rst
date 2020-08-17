.. _guide-minibatch:

Stochastic Training on Large Graphs
===================================

If we have a massive graph with, say, millions or even billions of nodes
or edges, usually full-graph training as described in
:ref:`guide-training`
would not work. Consider an :math:`L`-layer graph convolutional network
with hidden state size :math:`H` running on an :math:`N`-node graph.
Storing the intermediate hidden states requires :math:`O(NLH)` memory,
easily exceeding one GPU’s capacity with large :math:`N`.

This section provides a way to perform stochastic minibatch training,
where we do not have to fit the feature of all the nodes into GPU.

Overview of Neighborhood Sampling Approaches
--------------------------------------------

Neighborhood sampling methods generally work as the following. For each
gradient descent step, we select a minibatch of nodes whose final
representations at the :math:`L`-th layer are to be computed. We then
take all or some of their neighbors at the :math:`L-1` layer. This
process continues until we reach the input. This iterative process
builds the dependency graph starting from the output and working
backwards to the input, as the figure below shows:

.. figure:: https://i.imgur.com/Y0z0qcC.png
   :alt: Imgur

   Imgur

With this, one can save the workload and computation resources for
training a GNN on a large graph.

DGL provides a few neighborhood samplers and a pipeline for training a
GNN with neighborhood sampling, as well as ways to customize your
sampling strategies.

Training GNN for Node Classification with Neighborhood Sampling in DGL
----------------------------------------------------------------------

To make your model been trained stochastically, you need to do the
followings:

-  Define a neighborhood sampler.
-  Adapt your model for minibatch training.
-  Modify your training loop.

The following sub-subsections address these steps one by one.

.. _guide-minibatch-node-classification-sampler:

Define a neighborhood sampler and data loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL provides several neighborhood sampler classes that generates the
computation dependencies needed for each layer given the nodes we wish
to compute on.

The simplest neighborhood sampler is
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler`
which makes the node gather messages from all of its neighbors.

To use a sampler provided by DGL, one also need to combine it with
:class:`~dgl.dataloading.pytorch.NodeDataLoader`, which iterates
over a set of nodes in minibatches.

For example, the following code creates a PyTorch DataLoader that
iterates over the training node ID array ``train_nids`` in batches,
putting the list of generated blocks onto GPU.

.. code:: python

    import dgl
    import dgl.nn as dglnn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

Iterating over the DataLoader will yield a list of specially created
graphs representing the computation dependencies on each layer. They are
called *blocks* in DGL.

.. code:: python

    input_nodes, output_nodes, blocks = next(iter(dataloader))
    print(blocks)

The iterator generates three items at a time. ``input_nodes`` describe
the nodes needed to compute the representation of ``output_nodes``.
``blocks`` describe for each GNN layer which node representations are to
be computed as output, which node representations are needed as input,
and how does representation from the input nodes propagate to the output
nodes.

For a complete list of supported builtin samplers, please refer to the
:ref:`neighborhood sampler API reference <api-dataloading-neighbor-sampling>`.

If you wish to develop your own neighborhood sampler or you want a more
detailed explanation of the concept of blocks, please refer to
:ref:`guide-minibatch-customizing-neighborhood-sampler`.

.. _guide-minibatch-node-classification-model:

Adapt your model for minibatch training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If your message passing modules are all provided by DGL, the changes
required to adapt your model to minibatch training is minimal. Take a
multi-layer GCN as an example. If your model on full graph is
implemented as follows:

.. code:: python

    class TwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dglnn.GraphConv(in_features, hidden_features)
            self.conv2 = dglnn.GraphConv(hidden_features, out_features)
    
        def forward(self, g, x):
            x = F.relu(self.conv1(g, x))
            x = F.relu(self.conv2(g, x))
            return x

Then all you need is to replace ``g`` with ``blocks`` generated above.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
            self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
    
        def forward(self, blocks, x):
            x = F.relu(self.conv1(blocks[0], x))
            x = F.relu(self.conv2(blocks[1], x))
            return x

The DGL ``GraphConv`` modules above accepts an element in ``blocks``
generated by the data loader as an argument.

:ref:`The API reference of each NN module <apinn>` will tell you
whether it supports accepting a block as an argument.

If you wish to use your own message passing module, please refer to
:ref:`guide-minibatch-custom-gnn-module`.

Training Loop
~~~~~~~~~~~~~

The training loop simply consists of iterating over the dataset with the
customized batching iterator. During each iteration that yields a list
of blocks, we:

1. Load the node features corresponding to the input nodes onto GPU. The
   node features can be stored in either memory or external storage.
   Note that we only need to load the input nodes’ features, as opposed
   to load the features of all nodes as in full graph training.
   
   If the features are stored in ``g.ndata``, then the features can be loaded
   by accessing the features in ``blocks[0].srcdata``, the features of
   input nodes of the first block, which is identical to all the
   necessary nodes needed for computing the final representations.

2. Feed the list of blocks and the input node features to the multilayer
   GNN and get the outputs.

3. Load the node labels corresponding to the output nodes onto GPU.
   Similarly, the node labels can be stored in either memory or external
   storage. Again, note that we only need to load the output nodes’
   labels, as opposed to load the labels of all nodes as in full graph
   training.
   
   If the features are stored in ``g.ndata``, then the labels
   can be loaded by accessing the features in ``blocks[-1].srcdata``,
   the features of output nodes of the last block, which is identical to
   the nodes we wish to compute the final representation.

4. Compute the loss and backpropagate.

.. code:: python

    model = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata['features']
        output_labels = blocks[-1].dstdata['label']
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

DGL provides an end-to-end stochastic training example `GraphSAGE
implementation <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling.py>`__.

For heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~~~~~

Training a graph neural network for node classification on heterogeneous
graph is similar.

For instance, we have previously seen
:ref:`how to train a 2-layer RGCN on full graph <guide-training-rgcn-node-classification>`.
The code for RGCN implementation on minibatch training looks very
similar to that (with self-loops, non-linearity and basis decomposition
removed for simplicity):

.. code:: python

    class StochasticTwoLayerRGCN(nn.Module):
        def __init__(self, in_feat, hidden_feat, out_feat):
            super().__init__()
            self.conv1 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                    for rel in rel_names
                })
            self.conv2 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                    for rel in rel_names
                })
    
        def forward(self, blocks, x):
            x = self.conv1(blocks[0], x)
            x = self.conv2(blocks[1], x)
            return x

Some of the samplers provided by DGL also support heterogeneous graphs.
For example, one can still use the provided
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` class and
:class:`~dgl.dataloading.pytorch.NodeDataLoader` class for
stochastic training. For full-neighbor sampling, the only difference
would be that you would specify a dictionary of node
types and node IDs for the training set.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

The training loop is almost the same as that of homogeneous graphs,
except for the implementation of ``compute_loss`` that will take in two
dictionaries of node types and predictions here.

.. code:: python

    model = StochasticTwoLayerRGCN(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata     # returns a dict
        output_labels = blocks[-1].dstdata     # returns a dict
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

DGL provides an end-to-end stochastic training example `RGCN
implementation <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py>`__.

Training GNN for Edge Classification with Neighborhood Sampling in DGL
----------------------------------------------------------------------

Training for edge classification/regression is somewhat similar to that
of node classification/regression with several notable differences.

Define a neighborhood sampler and data loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the
:ref:`same neighborhood samplers as node classification <guide-minibatch-node-classification-sampler>`.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

To use the neighborhood sampler provided by DGL for edge classification,
one need to instead combine it with
:class:`~dgl.dataloading.pytorch.EdgeDataLoader`, which iterates
over a set of edges in minibatches, yielding the subgraph induced by the
edge minibatch and ``blocks`` to be consumed by the module above.

For example, the following code creates a PyTorch DataLoader that
iterates over the training edge ID array ``train_eids`` in batches,
putting the list of generated blocks onto GPU.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

For a complete list of supported builtin samplers, please refer to the
:ref:`neighborhood sampler API reference <api-dataloading-neighbor-sampling>`.

If you wish to develop your own neighborhood sampler or you want a more
detailed explanation of the concept of blocks, please refer to
:ref:`guide-minibatch-customizing-neighborhood-sampler`.

Removing edges in the minibatch from the original graph for neighbor sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training edge classification models, sometimes you wish to remove
the edges appearing in the training data from the computation dependency
as if they never existed. Otherwise, the model will “know” the fact that
an edge exists between the two nodes, and potentially use it for
advantage.

Therefore in edge classification you sometimes would like to exclude the
edges sampled in the minibatch from the original graph for neighborhood
sampling, as well as the reverse edges of the sampled edges on an
undirected graph. You can specify ``exclude='reverse'`` in instantiation
of :class:`~dgl.dataloading.pytorch.EdgeDataLoader`, with the mapping of the edge
IDs to their reverse edges IDs.  Usually doing so will lead to much slower
sampling process due to locating the reverse edges involving in the minibatch
and removing them.

.. code:: python

    n_edges = g.number_of_edges()
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
    
        # The following two arguments are specifically for excluding the minibatch
        # edges and their reverse edges from the original graph for neighborhood
        # sampling.
        exclude='reverse',
        reverse_eids=torch.cat([
            torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]),
    
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

Adapt your model for minibatch training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The edge classification model usually consists of two parts:

-  One part that obtains the representation of incident nodes.
-  The other part that computes the edge score from the incident node
   representations.

The former part is exactly the same as
:ref:`that from node classification <guide-minibatch-node-classification-model>`
and we can simply reuse it. The input is still the list of
blocks generated from a data loader provided by DGL, as well as the
input features.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dglnn.GraphConv(in_features, hidden_features)
            self.conv2 = dglnn.GraphConv(hidden_features, out_features)
    
        def forward(self, blocks, x):
            x = F.relu(self.conv1(blocks[0], x))
            x = F.relu(self.conv2(blocks[1], x))
            return x

The input to the latter part is usually the output from the
former part, as well as the subgraph of the original graph induced by the
edges in the minibatch. The subgraph is yielded from the same data
loader. One can call :meth:`dgl.DGLGraph.apply_edges` to compute the
scores on the edges with the edge subgraph.

The following code shows an example of predicting scores on the edges by
concatenating the incident node features and projecting it with a dense
layer.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def apply_edges(self, edges):
            data = torch.cat([edges.src['x'], edges.dst['x']])
            return {'score': self.W(data)}
    
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                edge_subgraph.apply_edges(self.apply_edges)
                return edge_subgraph.edata['score']

The entire model will take the list of blocks and the edge subgraph
generated by the data loader, as well as the input node features as
follows:

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes):
            super().__init__()
            self.gcn = StochasticTwoLayerGCN(
                in_features, hidden_features, out_features)
            self.predictor = ScorePredictor(num_classes, out_features)
    
        def forward(self, edge_subgraph, blocks, x):
            x = self.gcn(blocks, x)
            return self.predictor(edge_subgraph, x)

DGL ensures that that the nodes in the edge subgraph are the same as the
output nodes of the last block in the generated list of blocks.

Training Loop
~~~~~~~~~~~~~

The training loop is very similar to node classification. You can
iterate over the dataloader and get a subgraph induced by the edges in
the minibatch, as well as the list of blocks necessary for computing
their incident node representations.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        edge_labels = edge_subgraph.edata['labels']
        edge_predictions = model(edge_subgraph, blocks, input_features)
        loss = compute_loss(edge_labels, edge_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

For heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~~~~~

The models computing the node representations on heterogeneous graphs
can also be used for computing incident node representations for edge
classification/regression.

.. code:: python

    class StochasticTwoLayerRGCN(nn.Module):
        def __init__(self, in_feat, hidden_feat, out_feat):
            super().__init__()
            self.conv1 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                    for rel in rel_names
                })
            self.conv2 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                    for rel in rel_names
                })
    
        def forward(self, blocks, x):
            x = self.conv1(blocks[0], x)
            x = self.conv2(blocks[1], x)
            return x

For score prediction, the only implementation difference between the
homogeneous graph and the heterogeneous graph is that we are looping
over the edge types for :meth:`~dgl.DGLGraph.apply_edges`.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def apply_edges(self, edges):
            data = torch.cat([edges.src['x'], edges.dst['x']])
            return {'score': self.W(data)}
    
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(self.apply_edges, etype=etype)
                return edge_subgraph.edata['score']

Data loader definition is also very similar to that of node
classification. The only difference is that you need
:class:`~dgl.dataloading.pytorch.EdgeDataLoader` instead of
:class:`~dgl.dataloading.pytorch.NodeDataLoader`, and you will be supplying a
dictionary of edge types and edge ID tensors instead of a dictionary of
node types and node ID tensors.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

Things become a little different if you wish to exclude the reverse
edges on heterogeneous graphs. On heterogeneous graphs, reverse edges
usually have a different edge type from the edges themselves, in order
to differentiate the “forward” and “backward” relationships (e.g.
``follow`` and ``followed by`` are reverse relations of each other,
``purchase`` and ``purchased by`` are reverse relations of each other,
etc.).

If each edge in a type has a reverse edge with the same ID in another
type, you can specify the mapping between edge types and their reverse
types. The way to exclude the edges in the minibatch as well as their
reverse edges then goes as follows.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
    
        # The following two arguments are specifically for excluding the minibatch
        # edges and their reverse edges from the original graph for neighborhood
        # sampling.
        exclude='reverse_types',
        reverse_etypes={'follow': 'followed by', 'followed by': 'follow',
                        'purchase': 'purchased by', 'purchased by': 'purchase'}
    
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

The training loop is again almost the same as that on homogeneous graph,
except for the implementation of ``compute_loss`` that will take in two
dictionaries of node types and predictions here.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        edge_labels = edge_subgraph.edata['labels']
        edge_predictions = model(edge_subgraph, blocks, input_features)
        loss = compute_loss(edge_labels, edge_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

`GCMC <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
is an example of edge classification on a bipartite graph.

Training GNN for Link Prediction with Neighborhood Sampling in DGL
------------------------------------------------------------------

Define a neighborhood sampler and data loader with negative sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can still use the same neighborhood sampler as the one in node/edge
classification.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

:class:`~dgl.dataloading.pytorch.EdgeDataLoader` in DGL also
supports generating negative samples for link prediction. To do so, you
need to provide the negative sampling function.
:class:`~dgl.dataloading.negative_sampler.Uniform` is a
function that does uniform sampling. For each source node of an edge, it
samples ``k`` negative destination nodes.

The following data loader will pick 5 negative destination nodes
uniformly for each source node of an edge.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

For the builtin negative samplers please see :ref:`api-dataloading-negative-sampling`.

You can also give your own negative sampler function, as long as it
takes in the original graph ``g`` and the minibatch edge ID array
``eid``, and returns a pair of source ID arrays and destination ID
arrays.

The following gives an example of custom negative sampler that samples
negative destination nodes according to a probability distribution
proportional to a power of degrees.

.. code:: python

    class NegativeSampler(object):
        def __init__(self, g, k):
            # caches the probability distribution
            self.weights = g.in_degrees().float() ** 0.75
            self.k = k
    
        def __call__(self, g, eids):
            src, _ = g.find_edges(eids)
            src = src.repeat_interleave(self.k)
            dst = self.weights.multinomial(len(src), replacement=True)
            return src, dst
    
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler,
        negative_sampler=NegativeSampler(g, 5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

Adapt your model for minibatch training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As explained in :ref:`guide-training-link-prediction`, link prediction is trained
via comparing the score of an edge (positive example) against a
non-existent edge (negative example). To compute the scores of edges you
can reuse the node representation computation model you have seen in
edge classification/regression.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
            self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
    
        def forward(self, blocks, x):
            x = F.relu(self.conv1(blocks[0], x))
            x = F.relu(self.conv2(blocks[1], x))
            return x

For score prediction, since you only need to predict a scalar score for
each edge instead of a probability distribution, this example shows how
to compute a score with a dot product of incident node representations.

.. code:: python

    class ScorePredictor(nn.Module):
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
                return edge_subgraph.edata['score']

When a negative sampler is provided, DGL’s data loader will generate
three items per minibatch:

-  A positive graph containing all the edges sampled in the minibatch.
-  A negative graph containing all the non-existent edges generated by
   the negative sampler.
-  A list of blocks generated by the neighborhood sampler.

So one can define the link prediction model as follows that takes in the
three items as well as the input features.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.gcn = StochasticTwoLayerGCN(
                in_features, hidden_features, out_features)
    
        def forward(self, positive_graph, negative_graph, blocks, x):
            x = self.gcn(blocks, x)
            pos_score = self.predictor(positive_graph, x)
            neg_score = self.predictor(negative_graph, x)
            return pos_score, neg_score

Training loop
~~~~~~~~~~~~~

The training loop simply involves iterating over the data loader and
feeding in the graphs as well as the input features to the model defined
above.

.. code:: python

    model = Model(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        pos_score, neg_score = model(positive_graph, blocks, input_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()

DGL provides the
`unsupervised learning GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling_unsupervised.py>`__
that shows an example of link prediction on homogeneous graphs.

For heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~~~~~

The models computing the node representations on heterogeneous graphs
can also be used for computing incident node representations for edge
classification/regression.

.. code:: python

    class StochasticTwoLayerRGCN(nn.Module):
        def __init__(self, in_feat, hidden_feat, out_feat):
            super().__init__()
            self.conv1 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                    for rel in rel_names
                })
            self.conv2 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                    for rel in rel_names
                })
    
        def forward(self, blocks, x):
            x = self.conv1(blocks[0], x)
            x = self.conv2(blocks[1], x)
            return x

For score prediction, the only implementation difference between the
homogeneous graph and the heterogeneous graph is that we are looping
over the edge types for :meth:`dgl.DGLGraph.apply_edges`.

.. code:: python

    class ScorePredictor(nn.Module):
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(
                        dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
                return edge_subgraph.edata['score']

Data loader definition is also very similar to that of edge
classification/regression. The only difference is that you need to give
the negative sampler and you will be supplying a dictionary of edge
types and edge ID tensors instead of a dictionary of node types and node
ID tensors.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

If you want to give your own negative sampling function, the function
should take in the original graph and the dictionary of edge types and
edge ID tensors. It should return a dictionary of edge types and
source-destination array pairs. An example is given as follows:

.. code:: python

    class NegativeSampler(object):
        def __init__(self, g, k):
            # caches the probability distribution
            self.weights = {
                etype: g.in_degrees(etype=etype).float() ** 0.75
                for etype in g.canonical_etypes}
            self.k = k
    
        def __call__(self, g, eids_dict):
            result_dict = {}
            for etype, eids in eids_dict.items():
                src, _ = g.find_edges(eids, etype=etype)
                src = src.repeat_interleave(self.k)
                dst = self.weights.multinomial(len(src), replacement=True)
                result_dict[etype] = (src, dst)
            return result_dict
    
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        negative_sampler=negative_sampler=NegativeSampler(g, 5),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

The training loop is again almost the same as that on homogeneous graph,
except for the implementation of ``compute_loss`` that will take in two
dictionaries of node types and predictions here.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        edge_labels = edge_subgraph.edata['labels']
        edge_predictions = model(edge_subgraph, blocks, input_features)
        loss = compute_loss(edge_labels, edge_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()


.. _guide-minibatch-customizing-neighborhood-sampler:

Customizing Neighborhood Sampler
--------------------------------

Although DGL provides some neighborhood sampling strategies, sometimes
users would want to write their own sampling strategy. This section
explains how to write your own strategy and plug it into your stochastic
GNN training framework.

Recall that in `How Powerful are Graph Neural
Networks <https://arxiv.org/pdf/1810.00826.pdf>`__, the definition of message
passing is:

.. math::


   \begin{gathered}
     \boldsymbol{a}_v^{(l)} = \rho^{(l)} \left(
       \left\lbrace
         \boldsymbol{h}_u^{(l-1)} : u \in \mathcal{N} \left( v \right)
       \right\rbrace
     \right)
   \\
     \boldsymbol{h}_v^{(l)} = \phi^{(l)} \left(
       \boldsymbol{h}_v^{(l-1)}, \boldsymbol{a}_v^{(l)}
     \right)
   \end{gathered}

where :math:`\rho^{(l)}` and :math:`\phi^{(l)}` are parameterized
functions, and :math:`\mathcal{N}(v)` is defined as the set of
predecessors (or *neighbors* if the graph is undirected) of :math:`v` on graph
:math:`\mathcal{G}`.

For instance, to perform a message passing for updating the red node in
the following graph:

.. figure:: https://i.imgur.com/xYPtaoy.png
   :alt: Imgur

   Imgur

One needs to aggregate the node features of its neighbors, shown as
green nodes:

.. figure:: https://i.imgur.com/OuvExp1.png
   :alt: Imgur

   Imgur

Neighborhood sampling with pencil and paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We then consider how multi-layer message passing works for computing the
output of a single node. In the following text we refer to the nodes
whose GNN outputs are to be computed as *seed nodes*.

.. code:: python

    import torch
    import dgl
    
    src = torch.LongTensor(
        [0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10,
         1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11])
    dst = torch.LongTensor(
        [1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11,
         0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10])
    g = dgl.graph((src, dst))
    g.ndata['x'] = torch.randn(12, 5)
    g.ndata['y'] = torch.randn(12, 1)

Finding the message passing dependency
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider computing with a 2-layer GNN the output of the seed node 8,
colored red, in the following graph:

.. figure:: https://i.imgur.com/xYPtaoy.png
   :alt: Imgur

   Imgur

By the formulation:

.. math::


   \begin{gathered}
     \boldsymbol{a}_8^{(2)} = \rho^{(2)} \left(
       \left\lbrace
         \boldsymbol{h}_u^{(1)} : u \in \mathcal{N} \left( 8 \right)
       \right\rbrace
     \right) = \rho^{(2)} \left(
       \left\lbrace
         \boldsymbol{h}_4^{(1)}, \boldsymbol{h}_5^{(1)},
         \boldsymbol{h}_7^{(1)}, \boldsymbol{h}_{11}^{(1)}
       \right\rbrace
     \right)
   \\
     \boldsymbol{h}_8^{(2)} = \phi^{(2)} \left(
       \boldsymbol{h}_8^{(1)}, \boldsymbol{a}_8^{(2)}
     \right)
   \end{gathered}

We can tell from the formulation that to compute
:math:`\boldsymbol{h}_8^{(2)}` we need messages from node 4, 5, 7 and 11
(colored green) along the edges visualized below.

.. figure:: https://i.imgur.com/Gwjz05H.png
   :alt: Imgur

   Imgur

This graph contains all the nodes in the original graph but only the
edges necessary for message passing to the given output nodes. We call
that the *frontier* of the second GNN layer for the red node 8.

Several functions can be used for generating frontiers. For instance,
:func:`dgl.in_subgraph()` is a function that induces a
subgraph by including all the nodes in the original graph, but only all
the incoming edges of the given nodes. You can use that as a frontier
for message passing along all the incoming edges.

.. code:: python

    frontier = dgl.in_subgraph(g, [8])
    print(frontier.all_edges())

For a concrete list, please refer to :ref:`api-subgraph-extraction` and
:ref:`api-sampling`.

Technically, any graph that has the same set of nodes as the original
graph can serve as a frontier. This serves as the basis for
:ref:`guide-minibatch-customizing-neighborhood-sampler-impl`.

The Bipartite Structure for Multi-layer Minibatch Message Passing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, to compute :math:`\boldsymbol{h}_8^{(2)}` from
:math:`\boldsymbol{h}_\cdot^{(1)}`, we cannot simply perform message
passing on the frontier directly, because it still contains all the
nodes from the original graph. Namely, we only need nodes 4, 5, 7, 8,
and 11 (green and red nodes) as input, as well as node 8 (red node) as output.
Since the number of nodes
for input and output is different, we need to perform message passing on
a small, bipartite-structured graph instead. We call such a
bipartite-structured graph that only contains the necessary input nodes
and output nodes a *block*. The following figure shows the block of the
second GNN layer for node 8.

.. figure:: https://i.imgur.com/stB2UlR.png
   :alt: Imgur

   Imgur

Note that the output nodes also appear in the input nodes. The reason is
that representations of output nodes from the previous layer are needed
for feature combination after message passing (i.e. :math:`\phi^{(2)}`).

DGL provides :func:`dgl.to_block` to convert any frontier
to a block where the first argument specifies the frontier and the
second argument specifies the output nodes. For instance, the frontier
above can be converted to a block with output node 8 with the code as
follows.

.. code:: python

    output_nodes = torch.LongTensor([8])
    block = dgl.to_block(frontier, output_nodes)

To find the number of input nodes and output nodes of a given node type,
one can use :meth:`dgl.DGLGraph.number_of_src_nodes` and
:meth:`dgl.DGLGraph.number_of_dst_nodes` methods.

.. code:: python

    num_input_nodes, num_output_nodes = block.number_of_src_nodes(), block.number_of_dst_nodes()
    print(num_input_nodes, num_output_nodes)

The block’s input node features can be accessed via member
:attr:`dgl.DGLGraph.srcdata` and :attr:`dgl.DGLGraph.srcnodes`, and
its output node features can be accessed via member
:attr:`dgl.DGLGraph.dstdata` and :attr:`dgl.DGLGraph.dstnodes`. The
syntax of ``srcdata``/``dstdata`` and ``srcnodes``/``dstnodes`` are
identical to :attr:`dgl.DGLGraph.ndata` and
:attr:`dgl.DGLGraph.nodes` in normal graphs.

.. code:: python

    block.srcdata['h'] = torch.randn(num_input_nodes, 5)
    block.dstdata['h'] = torch.randn(num_output_nodes, 5)

If a block is converted from a frontier, which is in turn converted from
a graph, one can directly read the feature of the block’s input and
output nodes via

.. code:: python

    print(block.srcdata['x'])
    print(block.dstdata['y'])

.. raw:: html

   <div class="alert alert-info">

::

   <b>ID Mappings</b>

The original node IDs of the input nodes and output nodes in the block
can be found as the feature ``dgl.NID``, and the mapping from the
block’s edge IDs to the input frontier’s edge IDs can be found as the
feature ``dgl.EID``.

.. raw:: html

   </div>

**Output Nodes**

DGL ensures that the output nodes of a block will always appear in the
input nodes. The output nodes will always index firstly in the input
nodes.

.. code:: python

    input_nodes = block.srcdata[dgl.NID]
    output_nodes = block.dstdata[dgl.NID]
    assert torch.equal(input_nodes[:len(output_nodes)], output_nodes)

As a result, the output nodes must cover all nodes that are the
destination of an edge in the frontier.

For example, consider the following frontier

.. figure:: https://i.imgur.com/g5Ptbj7.png
   :alt: Imgur

   Imgur

where the red and green nodes (i.e. node 4, 5, 7, 8, and 11) are all
nodes that is a destination of an edge. Then the following code will
raise an error because the output nodes did not cover all those nodes.

.. code:: python

    dgl.to_block(frontier2, torch.LongTensor([4, 5]))   # ERROR

However, the output nodes can have more nodes than above. In this case,
we will have isolated nodes that do not have any edge connecting to it.
The isolated nodes will be included in both input nodes and output
nodes.

.. code:: python

    # Node 3 is an isolated node that do not have any edge pointing to it.
    block3 = dgl.to_block(frontier2, torch.LongTensor([4, 5, 7, 8, 11, 3]))
    print(block3.srcdata[dgl.NID])
    print(block3.dstdata[dgl.NID])

Heterogeneous Graphs
^^^^^^^^^^^^^^^^^^^^

Blocks also work on heterogeneous graphs. Let’s say that we have the
following frontier:

.. code:: python

    hetero_frontier = dgl.heterograph({
        ('user', 'follow', 'user'): ([1, 3, 7], [3, 6, 8]),
        ('user', 'play', 'game'): ([5, 5, 4], [6, 6, 2]),
        ('game', 'played-by', 'user'): ([2], [6])
    }, num_nodes_dict={'user': 10, 'game': 10})

One can also create a block with output nodes User #3, #6, and #8, as
well as Game #2 and #6.

.. code:: python

    hetero_block = dgl.to_block(hetero_frontier, {'user': [3, 6, 8], 'block': [2, 6]})

One can also get the input nodes and output nodes by type:

.. code:: python

    # input users and games
    print(hetero_block.srcnodes['user'].data[dgl.NID], hetero_block.srcnodes['game'].data[dgl.NID])
    # output users and games
    print(hetero_block.dstnodes['user'].data[dgl.NID], hetero_block.dstnodes['game'].data[dgl.NID])


.. _guide-minibatch-customizing-neighborhood-sampler-impl:

Implementing a Custom Neighbor Sampler
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall that the following code performs neighbor sampling for node
classification.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

To implement your own neighborhood sampling strategy, you basically
replace the ``sampler`` object with your own. To do that, let’s first
see what :class:`~dgl.dataloading.dataloader.BlockSampler`, the parent class of
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler`, is.

:class:`~dgl.dataloading.dataloader.BlockSampler` is responsible for
generating the list of blocks starting from the last layer, with method
:meth:`~dgl.dataloading.dataloader.BlockSampler.sample_blocks`. The default implementation of
``sample_blocks`` is to iterate backwards, generating the frontiers and
converting them to blocks.

Therefore, for neighborhood sampling, **you only need to implement
the**\ :meth:`~dgl.dataloading.dataloader.BlockSampler.sample_frontier`\ **method**. Given which
layer the sampler is generating frontier for, as well as the original
graph and the nodes to compute representations, this method is
responsible for generating a frontier for them.

Meanwhile, you also need to pass how many GNN layers you have to the
parent class.

For example, the implementation of
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` can
go as follows.

.. code:: python

    class MultiLayerFullNeighborSampler(dgl.dataloading.BlockSampler):
        def __init__(self, n_layers):
            super().__init__(n_layers)
    
        def sample_frontier(self, block_id, g, seed_nodes):
            frontier = dgl.in_subgraph(g, seed_nodes)
            return frontier

:class:`dgl.dataloading.neighbor.MultiLayerNeighborSampler`, a more
complicated neighbor sampler class that allows you to sample a small
number of neighbors to gather message for each node, goes as follows.

.. code:: python

    class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
        def __init__(self, fanouts):
            super().__init__(len(fanouts))
    
            self.fanouts = fanouts
    
        def sample_frontier(self, block_id, g, seed_nodes):
            fanout = self.fanouts[block_id]
            if fanout is None:
                frontier = dgl.in_subgraph(g, seed_nodes)
            else:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            return frontier

Although the functions above can generate a frontier, any graph that has
the same nodes as the original graph can serve as a frontier.

For example, if one want to randomly drop inbound edges to the seed
nodes with a probability, one can simply define the sampler as follows:

.. code:: python

    class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
        def __init__(self, p, n_layers):
            super().__init__()
    
            self.n_layers = n_layers
            self.p = p
    
        def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
            # Get all inbound edges to `seed_nodes`
            src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
            # Randomly select edges with a probability of p
            mask = torch.zeros_like(src).bernoulli_(self.p)
            src = src[mask]
            dst = dst[mask]
            # Return a new graph with the same nodes as the original graph as a
            # frontier
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
            return frontier
    
        def __len__(self):
            return self.n_layers

After implementing your sampler, you can create a data loader that takes
in your sampler and it will keep generating lists of blocks while
iterating over the seed nodes as usual.

.. code:: python

    sampler = MultiLayerDropoutSampler(0.5, 2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    
    model = StochasticTwoLayerRGCN(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata     # returns a dict
        output_labels = blocks[-1].dstdata     # returns a dict
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

Heterogeneous Graphs
^^^^^^^^^^^^^^^^^^^^

Generating a frontier for a heterogeneous graph is nothing different
than that for a homogeneous graph. Just make the returned graph have the
same nodes as the original graph, and it should work fine. For example,
we can rewrite the ``MultiLayerDropoutSampler`` above to iterate over
all edge types, so that it can work on heterogeneous graphs as well.

.. code:: python

    class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
        def __init__(self, p, n_layers):
            super().__init__()
    
            self.n_layers = n_layers
            self.p = p
    
        def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
            # Get all inbound edges to `seed_nodes`
            sg = dgl.in_subgraph(g, seed_nodes)
    
            new_edges_masks = {}
            # Iterate over all edge types
            for etype in sg.canonical_etypes:
                edge_mask = torch.zeros(sg.number_of_edges(etype))
                edge_mask.bernoulli_(self.p)
                new_edges_masks[etype] = edge_mask.bool()
    
            # Return a new graph with the same nodes as the original graph as a
            # frontier
            frontier = dgl.edge_subgraph(new_edge_masks, preserve_nodes=True)
            return frontier
    
        def __len__(self):
            return self.n_layers


.. _guide-minibatch-custom-gnn-module:

Implementing Custom GNN Module with Blocks
------------------------------------------

If you were familiar with how to write a custom GNN module for updating
the entire graph for homogeneous or heterogeneous graphs (see
:ref:`guide-nn`), the code for computing on
blocks is similar, with the exception that the nodes are divided into
input nodes and output nodes.

For example, consider the following custom graph convolution module
code. Note that it is not necessarily among the most efficient implementations
- they only serve for an example of how a custom GNN module could look
like.

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            with g.local_scope():
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))

If you have a custom message passing NN module for the full graph, and
you would like to make it work for blocks, you only need to rewrite the
forward function as follows. Note that the corresponding statements from
the full-graph implementation are commented; you can compare the
original statements with the new statements.

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        # h is now a pair of feature tensors for input and output nodes, instead of
        # a single feature tensor.
        # def forward(self, g, h):
        def forward(self, block, h):
            # with g.local_scope():
            with block.local_scope():
                # g.ndata['h'] = h
                h_src = h
                h_dst = h[:block.number_of_dst_nodes()]
                block.srcdata['h'] = h_src
                block.dstdata['h'] = h_dst
    
                # g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
    
                # return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
                return self.W(torch.cat(
                    [block.dstdata['h'], block.dstdata['h_neigh']], 1))

In general, you need to do the following to make your NN module work for
blocks.

-  Obtain the features for output nodes from the input features by
   slicing the first few rows. The number of rows can be obtained by
   :meth:`block.number_of_dst_nodes <dgl.DGLGraph.number_of_dst_nodes>`.
-  Replace
   :attr:`g.ndata <dgl.DGLGraph.ndata>` with either
   :attr:`block.srcdata <dgl.DGLGraph.srcdata>` for features on input nodes or
   :attr:`block.dstdata <dgl.DGLGraph.dstdata>` for features on output nodes, if
   the original graph has only one node type.
-  Replace
   :attr:`g.nodes <dgl.DGLGraph.nodes>` with either
   :attr:`block.srcnodes <dgl.DGLGraph.srcnodes>` for features on input nodes or
   :attr:`block.dstnodes <dgl.DGLGraph.dstnodes>` for features on output nodes,
   if the original graph has multiple node types.
-  Replace
   :meth:`g.number_of_nodes <dgl.DGLGraph.number_of_nodes>` with either
   :meth:`block.number_of_src_nodes <dgl.DGLGraph.number_of_src_nodes>` or
   :meth:`block.number_of_dst_nodes <dgl.DGLGraph.number_of_dst_nodes>` for the number of
   input nodes or output nodes respectively.

Heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~

For heterogeneous graph the way of writing custom GNN modules is
similar. For instance, consider the following module that work on full
graph.

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    g.nodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.nodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.nodes[vtype].data['h_dst'] = g.nodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.nodes[vtype].data['h_neigh'])
                return {ntype: g.nodes[ntype].data['h_dst'] for ntype in g.ntypes}

For ``CustomHeteroGraphConv``, the principle is to replace ``g.nodes``
with ``g.srcnodes`` or ``g.dstnodes`` depend on whether the features
serve for input or output.

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    h_src, h_dst = h[ntype]
                    g.dstnodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.srcnodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.dstnodes[vtype].data['h_dst'] = \
                        g.dstnodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.dstnodes[vtype].data['h_neigh'])
                return {ntype: g.dstnodes[ntype].data['h_dst']
                        for ntype in g.ntypes}

Writing modules that work on homogeneous graphs, bipartite graphs, and blocks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All message passing modules in DGL work on homogeneous graphs,
unidirectional bipartite graphs (that have two node types and one edge
type), and a block with one edge type. Essentially, the input graph and
feature of a builtin DGL neural network module must satisfy either of
the following cases.

-  If the input feature is a pair of tensors, then the input graph must
   be unidirectional bipartite.
-  If the input feature is a single tensor and the input graph is a
   block, DGL will automatically set the feature on the output nodes as
   the first few rows of the input node features.
-  If the input feature must be a single tensor and the input graph is
   not a block, then the input graph must be homogeneous.

For example, the following is simplified from the PyTorch implementation
of :class:`dgl.nn.pytorch.conv.SAGEConv` (also available in MXNet and Tensorflow)
(removing normalization and dealing with only mean aggregation etc.).

.. code:: python

    import dgl.function as fn
    class SAGEConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            if isinstance(h, tuple):
                h_src, h_dst = h
            elif g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h
                 
            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            return F.relu(
                self.W(torch.cat([g.dstdata['h'], g.dstdata['h_neigh']], 1)))

:ref:`guide-nn` also provides a walkthrough on :class:`dgl.nn.pytorch.conv.SAGEConv`,
which works on unidirectional bipartite graphs, homogeneous graphs, and blocks.

Exact Offline Inference on Large Graphs
---------------------------------------

Both subgraph sampling and neighborhood sampling are to reduce the
memory and time consumption for training GNNs with GPUs. When performing
inference it is usually better to truly aggregate over all neighbors
instead to get rid of the randomness introduced by sampling. However,
full-graph forward propagation is usually infeasible on GPU due to
limited memory, and slow on CPU due to slow computation. This section
introduces the methodology of full-graph forward propagation with
limited GPU memory via minibatch and neighborhood sampling.

The inference algorithm is different from the training algorithm, as the
representations of all nodes should be computed layer by layer, starting
from the first layer. Specifically, for a particular layer, we need to
compute the output representations of all nodes from this GNN layer in
minibatches. The consequence is that the inference algorithm will have
an outer loop iterating over the layers, and an inner loop iterating
over the minibatches of nodes. In contrast, the training algorithm has
an outer loop iterating over the minibatches of nodes, and an inner loop
iterating over the layers for both neighborhood sampling and message
passing.

The following animation shows how the computation would look like (note
that for every layer only the first three minibatches are drawn).

.. figure:: https://i.imgur.com/rr1FG7S.gif
   :alt: Imgur

   Imgur

Implementing Offline Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider the two-layer GCN we have mentioned in Section 6.5.1. The way
to implement offline inference still involves using
```MultiLayerFullNeighborSampler`` <https://todo>`__, but sampling for
only one layer at a time. Note that offline inference is implemented as
a method of the GNN module because the computation on one layer depends
on how messages are aggregated and combined as well.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.hidden_features = hidden_features
            self.out_features = out_features
            self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
            self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
            self.n_layers = 2
    
        def forward(self, blocks, x):
            x_dst = x[:blocks[0].number_of_dst_nodes()]
            x = F.relu(self.conv1(blocks[0], (x, x_dst)))
            x_dst = x[:blocks[1].number_of_dst_nodes()]
            x = F.relu(self.conv2(blocks[1], (x, x_dst)))
            return x
    
        def inference(self, g, x, batch_size, device):
            """
            Offline inference with this module
            """
            # Compute representations layer by layer
            for l, layer in enumerate([self.conv1, self.conv2]):
                y = torch.zeros(g.number_of_nodes(),
                                self.hidden_features
                                if l != self.n_layers - 1
                                else self.out_features)
                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
                dataloader = dgl.dataloading.NodeDataLoader(
                    g, torch.arange(g.number_of_nodes()), sampler,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=False)
                
                # Within a layer, iterate over nodes in batches
                for input_nodes, output_nodes, blocks in dataloader:
                    block = blocks[0]
    
                    # Copy the features of necessary input nodes to GPU
                    h = x[input_nodes].to(device)
                    # Compute output.  Note that this computation is the same
                    # but only for a single layer.
                    h_dst = h[:block.number_of_dst_nodes()]
                    h = F.relu(layer(block, (h, h_dst)))
                    # Copy to output back to CPU.
                    y[output_nodes] = h.cpu()
    
            return y

Note that for the purpose of computing evaluation metric on the
validation set for model selection we usually don’t have to compute
exact offline inference. The reason is that we need to compute the
representation for every single node on every single layer, which is
usually very costly especially in the semi-supervised regime with a lot
of unlabeled data. Neighborhood sampling will work fine for model
selection and validation.

One can see
`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling.py>`__
and
`RGCN <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py>`__
for examples of offline inference.
