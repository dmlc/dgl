.. _guide-minibatch-edge-classification-sampler:

6.2 Training GNN for Edge Classification with Neighborhood Sampling
----------------------------------------------------------------------

:ref:`(中文版) <guide_cn-minibatch-edge-classification-sampler>`

Training for edge classification/regression is somewhat similar to that
of node classification/regression with several notable differences.

Define a neighborhood sampler and data loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use the
:ref:`same neighborhood samplers as node classification <guide-minibatch-node-classification-sampler>`.

.. code:: python

    datapipe = datapipe.sample_neighbor(g, [10, 10])
    # Or equivalently
    datapipe = dgl.graphbolt.NeighborSampler(datapipe, g, [10, 10])

The code for defining a data loader is also the same as that of node
classification. The only difference is that it iterates over the
edges(namely, node pairs) in the training set instead of the nodes.

.. code:: python

    import dgl.graphbolt as gb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = gb.SamplingGraph()
    seeds = torch.arange(0, 1000).reshape(-1, 2)
    labels = torch.randint(0, 2, (5,))
    train_set = gb.ItemSet((seeds, labels), names=("seeds", "labels"))
    datapipe = gb.ItemSampler(train_set, batch_size=128, shuffle=True)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    # Or equivalently:
    # datapipe = gb.NeighborSampler(datapipe, g, [10, 10])
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

Iterating over the DataLoader will yield :class:`~dgl.graphbolt.MiniBatch`
which contains a list of specially created graphs representing the computation
dependencies on each layer. You can access the *message flow graphs* (MFGs) via
`mini_batch.blocks`.

.. code:: python
    mini_batch = next(iter(dataloader))
    print(mini_batch.blocks)

.. note::

   See the :doc:`Stochastic Training Tutorial
   <../notebooks/stochastic_training/neighbor_sampling_overview.nblink>`__
   for the concept of message flow graph.

   If you wish to develop your own neighborhood sampler or you want a more
   detailed explanation of the concept of MFGs, please refer to
   :ref:`guide-minibatch-customizing-neighborhood-sampler`.

.. _guide-minibatch-edge-classification-sampler-exclude:

Removing edges in the minibatch from the original graph for neighbor sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When training edge classification models, sometimes you wish to remove
the edges appearing in the training data from the computation dependency
as if they never existed. Otherwise, the model will “know” the fact that
an edge exists between the two nodes, and potentially use it for
advantage.

Therefore in edge classification you sometimes would like to exclude the
seed edges as well as their reverse edges from the sampled minibatch.
You can use :func:`~dgl.graphbolt.exclude_seed_edges` alongside with
:class:`~dgl.graphbolt.MiniBatchTransformer` to achieve this.

.. code:: python

    import dgl.graphbolt as gb
    from functools import partial

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = gb.SamplingGraph()
    seeds = torch.arange(0, 1000).reshape(-1, 2)
    labels = torch.randint(0, 2, (5,))
    train_set = gb.ItemSet((seeds, labels), names=("seeds", "labels"))
    datapipe = gb.ItemSampler(train_set, batch_size=128, shuffle=True)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    exclude_seed_edges = partial(gb.exclude_seed_edges, include_reverse_edges=True)
    datapipe = datapipe.transform(exclude_seed_edges)
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)
    

Adapt your model for minibatch training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The edge classification model usually consists of two parts:

-  One part that obtains the representation of incident nodes.
-  The other part that computes the edge score from the incident node
   representations.

The former part is exactly the same as
:ref:`that from node classification <guide-minibatch-node-classification-model>`
and we can simply reuse it. The input is still the list of
MFGs generated from a data loader provided by DGL, as well as the
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
former part, as well as the subgraph(node pairs) of the original graph induced
by the edges in the minibatch. The subgraph is yielded from the same data
loader.

The following code shows an example of predicting scores on the edges by
concatenating the incident node features and projecting it with a dense layer.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def forward(self, seeds, x):
            src_x = x[seeds[:, 0]]
            dst_x = x[seeds[:, 1]]
            data = torch.cat([src_x, dst_x], 1)
            return self.W(data)


The entire model will take the list of MFGs and the edges generated by the data
loader, as well as the input node features as follows:

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes):
            super().__init__()
            self.gcn = StochasticTwoLayerGCN(
                in_features, hidden_features, out_features)
            self.predictor = ScorePredictor(num_classes, out_features)

        def forward(self, blocks, x, seeds):
            x = self.gcn(blocks, x)
            return self.predictor(seeds, x)

DGL ensures that that the nodes in the edge subgraph are the same as the
output nodes of the last MFG in the generated list of MFGs.

Training Loop
~~~~~~~~~~~~~

The training loop is very similar to node classification. You can
iterate over the dataloader and get a subgraph induced by the edges in
the minibatch, as well as the list of MFGs necessary for computing
their incident node representations.

.. code:: python

    import torch.nn.functional as F
    model = Model(in_features, hidden_features, out_features, num_classes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    for data in dataloader:
        blocks = data.blocks
        x = data.edge_features("feat")
        y_hat = model(data.blocks, x, data.compacted_seeds)
        loss = F.cross_entropy(data.labels, y_hat)
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
        def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
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
over the edge types.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def forward(self, seeds, x):
            scores = {}
            for etype in seeds.keys():
                src, dst = seeds[etype].T
                data = torch.cat([x[etype][src], x[etype][dst]], 1)
                scores[etype] = self.W(data)
            return scores

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes,
                     etypes):
            super().__init__()
            self.rgcn = StochasticTwoLayerRGCN(
                in_features, hidden_features, out_features, etypes)
            self.pred = ScorePredictor(num_classes, out_features)

        def forward(self, seeds, blocks, x):
            x = self.rgcn(blocks, x)
            return self.pred(seeds, x)

Data loader definition is almost identical to that of homogeneous graph. The
only difference is that the train_set is now an instance of
:class:`~dgl.graphbolt.HeteroItemSet` instead of :class:`~dgl.graphbolt.ItemSet`.

.. code:: python

    import dgl.graphbolt as gb

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    g = gb.SamplingGraph()
    seeds = torch.arange(0, 1000).reshape(-1, 2)
    labels = torch.randint(0, 3, (1000,))
    seeds_labels = {
        "user:like:item": gb.ItemSet(
            (seeds, labels), names=("seeds", "labels")
        ),
        "user:follow:user": gb.ItemSet(
            (seeds, labels), names=("seeds", "labels")
        ),
    }
    train_set = gb.HeteroItemSet(seeds_labels)
    datapipe = gb.ItemSampler(train_set, batch_size=128, shuffle=True)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.fetch_feature(
        feature, node_feature_keys={"item": ["feat"], "user": ["feat"]}
    )
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

Things become a little different if you wish to exclude the reverse
edges on heterogeneous graphs. On heterogeneous graphs, reverse edges
usually have a different edge type from the edges themselves, in order
to differentiate the “forward” and “backward” relationships (e.g.
``follow`` and ``followed_by`` are reverse relations of each other,
``like`` and ``liked_by`` are reverse relations of each other,
etc.).

If each edge in a type has a reverse edge with the same ID in another
type, you can specify the mapping between edge types and their reverse
types. The way to exclude the edges in the minibatch as well as their
reverse edges then goes as follows.

.. code:: python


    exclude_seed_edges = partial(
        gb.exclude_seed_edges,
        include_reverse_edges=True,
        reverse_etypes_mapping={
            "user:like:item": "item:liked_by:user",
            "user:follow:user": "user:followed_by:user",
        },
    )
    datapipe = datapipe.transform(exclude_seed_edges)


The training loop is again almost the same as that on homogeneous graph,
except for the implementation of ``compute_loss`` that will take in two
dictionaries of node types and predictions here.

.. code:: python

    import torch.nn.functional as F
    model = Model(in_features, hidden_features, out_features, num_classes, etypes)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())

    for data in dataloader:
        blocks = data.blocks
        x = data.edge_features(("user:like:item", "feat"))
        y_hat = model(data.blocks, x, data.compacted_seeds)
        loss = F.cross_entropy(data.labels, y_hat)
        opt.zero_grad()
        loss.backward()
        opt.step()

