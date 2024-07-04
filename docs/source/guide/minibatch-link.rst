.. _guide-minibatch-link-classification-sampler:

6.3 Training GNN for Link Prediction with Neighborhood Sampling
--------------------------------------------------------------------

:ref:`(中文版) <guide_cn-minibatch-link-classification-sampler>`

Define a data loader with neighbor and negative sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can still use the same data loader as the one in node/edge classification.
The only difference is that you need to add an additional stage
`negative sampling` before neighbor sampling stage. The following data loader
will pick 5 negative destination nodes uniformly for each source node of an
edge.

.. code:: python

    datapipe = datapipe.sample_uniform_negative(graph, 5)

The whole data loader pipeline is as follows:

.. code:: python

    datapipe = gb.ItemSampler(itemset, batch_size=1024, shuffle=True)
    datapipe = datapipe.sample_uniform_negative(graph, 5)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.transform(gb.exclude_seed_edges)
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)


For the details about the builtin uniform negative sampler please see
:class:`~dgl.graphbolt.UniformNegativeSampler`.

You can also give your own negative sampler function, as long as it inherits
from :class:`~dgl.graphbolt.NegativeSampler` and overrides the
:meth:`~dgl.graphbolt.NegativeSampler._sample_with_etype` method which takes in
the node pairs in minibatch, and returns the negative node pairs back.

The following gives an example of custom negative sampler that samples
negative destination nodes according to a probability distribution
proportional to a power of degrees.

.. code:: python

    @functional_datapipe("customized_sample_negative")
    class CustomizedNegativeSampler(dgl.graphbolt.NegativeSampler):
        def __init__(self, datapipe, k, node_degrees):
            super().__init__(datapipe, k)
            # caches the probability distribution
            self.weights = node_degrees ** 0.75
            self.k = k
    
        def _sample_with_etype(self, seeds, etype=None):
            src, _ = seeds.T
            src = src.repeat_interleave(self.k)
            dst = self.weights.multinomial(len(src), replacement=True)
            return src, dst

    datapipe = datapipe.customized_sample_negative(5, node_degrees)


Define a GraphSAGE model for minibatch training
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    class SAGE(nn.Module):
        def __init__(self, in_size, hidden_size):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
            self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
            self.hidden_size = hidden_size
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, blocks, x):
            hidden_x = x
            for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                hidden_x = layer(block, hidden_x)
                is_last_layer = layer_idx == len(self.layers) - 1
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
            return hidden_x


When a negative sampler is provided, the data loader will generate positive and
negative node pairs for each minibatch besides the *Message Flow Graphs* (MFGs).
Use `compacted_seeds` and `labels` to get compact node pairs and corresponding
labels.


Training loop
~~~~~~~~~~~~~

The training loop simply involves iterating over the data loader and
feeding in the graphs as well as the input features to the model defined
above.

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm.trange(args.epochs):
        model.train()
        total_loss = 0
        start_epoch_time = time.time()
        for step, data in enumerate(dataloader):
            # Unpack MiniBatch.
            compacted_seeds = data.compacted_seeds.T
            labels = data.labels
            node_feature = data.node_features["feat"]
            # Convert sampled subgraphs to DGL blocks.
            blocks = data.blocks

            # Get the embeddings of the input nodes.
            y = model(blocks, node_feature)
            logits = model.predictor(
                y[compacted_seeds[0]] * y[compacted_seeds[1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        end_epoch_time = time.time()


DGL provides the
`unsupervised learning GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/graphbolt/link_prediction.py>`__
that shows an example of link prediction on homogeneous graphs.

For heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~~~~~

The previous model could be easily extended to heterogeneous graphs. The only
difference is that you need to use :class:`~dgl.nn.HeteroGraphConv` to wrap
:class:`~dgl.nn.SAGEConv` according to edge types.

.. code:: python

    class SAGE(nn.Module):
        def __init__(self, in_size, hidden_size):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.HeteroGraphConv({
                    rel : dglnn.SAGEConv(in_size, hidden_size, "mean")
                    for rel in rel_names
                }))
            self.layers.append(dglnn.HeteroGraphConv({
                    rel : dglnn.SAGEConv(hidden_size, hidden_size, "mean")
                    for rel in rel_names
                }))
            self.layers.append(dglnn.HeteroGraphConv({
                    rel : dglnn.SAGEConv(hidden_size, hidden_size, "mean")
                    for rel in rel_names
                }))
            self.hidden_size = hidden_size
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, blocks, x):
            hidden_x = x
            for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
                hidden_x = layer(block, hidden_x)
                is_last_layer = layer_idx == len(self.layers) - 1
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
            return hidden_x


Data loader definition is also very similar to that for homogeneous graph. The
only difference is that you need to give edge types for feature fetching.

.. code:: python

    datapipe = gb.ItemSampler(itemset, batch_size=1024, shuffle=True)
    datapipe = datapipe.sample_uniform_negative(graph, 5)
    datapipe = datapipe.sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.transform(gb.exclude_seed_edges)
    datapipe = datapipe.fetch_feature(
        feature,
        node_feature_keys={"user": ["feat"], "item": ["feat"]}
    )
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

If you want to give your own negative sampling function, just inherit from the
:class:`~dgl.graphbolt.NegativeSampler` class and override the
:meth:`~dgl.graphbolt.NegativeSampler._sample_with_etype` method.

.. code:: python

    @functional_datapipe("customized_sample_negative")
    class CustomizedNegativeSampler(dgl.graphbolt.NegativeSampler):
        def __init__(self, datapipe, k, node_degrees):
            super().__init__(datapipe, k)
            # caches the probability distribution
            self.weights = {
                etype: node_degrees[etype] ** 0.75 for etype in node_degrees
            }
            self.k = k
    
        def _sample_with_etype(self, seeds, etype):
            src, _ = seeds.T
            src = src.repeat_interleave(self.k)
            dst = self.weights[etype].multinomial(len(src), replacement=True)
            return src, dst

    datapipe = datapipe.customized_sample_negative(5, node_degrees)


For heterogeneous graphs, node pairs are grouped by edge types. The training
loop is again almost the same as that on homogeneous graph, except for computing
loss on specific edge type.

.. code:: python

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    category = "user"
    for epoch in tqdm.trange(args.epochs):
        model.train()
        total_loss = 0
        start_epoch_time = time.time()
        for step, data in enumerate(dataloader):
            # Unpack MiniBatch.
            compacted_seeds = data.compacted_seeds
            labels = data.labels
            node_features = {
                ntype: data.node_features[(ntype, "feat")]
                for ntype in data.blocks[0].srctypes
            }
            # Convert sampled subgraphs to DGL blocks.
            blocks = data.blocks
            # Get the embeddings of the input nodes.
            y = model(blocks, node_feature)
            logits = model.predictor(
                y[category][compacted_pairs[category][:, 0]]
                * y[category][compacted_pairs[category][:, 1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(logits, labels[category])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        end_epoch_time = time.time()

