.. _guide-minibatch-customizing-neighborhood-sampler:

6.4 Implementing Custom Graph Samplers
----------------------------------------------

Implementing custom samplers involves subclassing the
:class:`dgl.graphbolt.SubgraphSampler` base class and implementing its abstract
:attr:`sample_subgraphs` method. The :attr:`sample_subgraphs` method should
take in seed nodes which are the nodes to sample neighbors from:

.. code:: python

    def sample_subgraphs(self, seed_nodes):
        return input_nodes, sampled_subgraphs

The method should return the input node IDs list and a list of subgraphs. Each
subgraph is a :class:`~dgl.graphbolt.SampledSubgraph` object.


Any other data that are required during sampling such as the graph structure,
fanout size, etc. should be passed to the sampler via the constructor.

The code below implements a classical neighbor sampler:

.. code:: python

    @functional_datapipe("customized_sample_neighbor")
    class CustomizedNeighborSampler(dgl.graphbolt.SubgraphSampler):
       def __init__(self, datapipe, graph, fanouts):
           super().__init__(datapipe)
           self.graph = graph
           self.fanouts = fanouts

       def sample_subgraphs(self, seed_nodes):
           subgs = []
           for fanout in reversed(self.fanouts):
               # Sample a fixed number of neighbors of the current seed nodes.
               input_nodes, sg = g.sample_neighbors(seed_nodes, fanout)
               subgs.insert(0, sg)
               seed_nodes = input_nodes
           return input_nodes, subgs

To use this sampler with :class:`~dgl.graphbolt.DataLoader`:

.. code:: python

    datapipe = gb.ItemSampler(train_set, batch_size=1024, shuffle=True)
    datapipe = datapipe.customized_sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.fetch_feature(feature, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

    for data in dataloader:
        input_features = data.node_features["feat"]
        output_labels = data.labels
        output_predictions = model(data.blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()


Sampler for Heterogeneous Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To write a sampler for heterogeneous graphs, one needs to be aware that
the argument `graph` is a heterogeneous graph while `seeds` could be a
dictionary of ID tensors. Most of DGL's graph sampling operators (e.g.,
the ``sample_neighbors`` and ``to_block`` functions in the above example) can
work on heterogeneous graph natively, so many samplers are automatically
ready for heterogeneous graph. For example, the above ``CustomizedNeighborSampler``
can be used on heterogeneous graphs:

.. code:: python

    import dgl.graphbolt as gb
    hg = gb.FusedCSCSamplingGraph()
    train_set = item_set = gb.HeteroItemSet(
        {
            "user": gb.ItemSet(
                (torch.arange(0, 5), torch.arange(5, 10)),
                names=("seeds", "labels"),
            ),
            "item": gb.ItemSet(
                (torch.arange(5, 10), torch.arange(10, 15)),
                names=("seeds", "labels"),
            ),
        }
    )
    datapipe = gb.ItemSampler(train_set, batch_size=1024, shuffle=True)
    datapipe = datapipe.customized_sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.fetch_feature(
        feature, node_feature_keys={"user": ["feat"], "item": ["feat"]}
    )
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)

    for data in dataloader:
        input_features = {
            ntype: data.node_features[(ntype, "feat")]
            for ntype in data.blocks[0].srctypes
        }
        output_labels = data.labels["user"]
        output_predictions = model(data.blocks, input_features)["user"]
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()


Exclude Edges After Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In some cases, we may want to exclude seed edges from the sampled subgraph. For
example, in link prediction tasks, we want to exclude the edges in the
training set from the sampled subgraph to prevent information leakage. To
do so, we need to add an additional datapipe right after sampling as follows:

.. code:: python

    datapipe = datapipe.customized_sample_neighbor(g, [10, 10]) # 2 layers.
    datapipe = datapipe.transform(gb.exclude_seed_edges)

Please check the API page of :func:`~dgl.graphbolt.exclude_seed_edges` for more
details.

The above API is based on :meth:`~dgl.graphbolt.SampledSubgrahp.exclude_edges`.
If you want to exclude edges from the sampled subgraph based on some other
criteria, you could write your own transform function. Please check the method
for reference.

You could also refer to examples in
`Link Prediction <https://github.com/dmlc/dgl/blob/master/examples/graphbolt/link_prediction.py>`__.
