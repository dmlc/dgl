.. _guide-minibatch-prefetching:

6.8 Feature Prefetching
-----------------------

In minibatch training of GNNs, especially with neighbor sampling approaches, we can see
that a large number of nodes' input features are necessary to compute a small number of
nodes' GNN representations.  Therefore, unlike traditional deep learning model training,
we can often observe that transferring features from CPU to GPU takes a majority of
time.

DGL thus introduces *feature prefetching* in dataloading pipeline that allows the transfer
from CPU to GPU to overlap with the model computation and sampling.

Enabling Prefetching with DGL's Builtin Samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All the DGL samplers in :ref:`api-dataloading` allows enabling prefetching of specific
node and edge data from the graph with arguments like :attr:`prefetch_node_feats`.  The
documentation describes which part of the sampled MFGs or subgraphs the prefetched
data will go into.

For example, with :class:`dgl.dataloading.NeighborSampler`, the following will prefetch
the node data named ``feat`` for the first layer's MFG's ``srcnodes``, as well as the
node data named ``label`` for the last layer's MFG's ``dstnodes``:

.. code:: python

   sampler = dgl.dataloading.NeighborSampler(
       [15, 10, 5], prefetch_node_feats=['feat'], prefetch_labels=['label'])

Enabling Prefetching in Custom Samplers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Prefetching can also be enabled with custom samplers, as long as the returned subgraphs
or MFGs have :class:`dgl.dataloading.LazyFeature` attached.

A :class:`dgl.dataloading.LazyFeature` works as a placeholder that tells the DGL dataloaders which
node/edge data to prefetch, and where DGL dataloaders should put them.  It takes in
a single string argument representing which node/edge data to prefetch, depending
on whether the :class:`dgl.dataloading.LazyFeature` object is assigned to
``ndata``/``srcdata``/``dstdata`` or ``edata``.  The DataLoader will inspect each object
returned by the sampler, and replace the :class:`dgl.dataloading.LazyFeature` objects with the
node and edge data retrieved from the original graph.

Consider an example where the sampler returns a subgraph induced by the seed nodes
sampled in the minibatch:

.. code:: python

   class ExampleSubgraphSampler(object):
       def sample(self, g, seed_nodes):
           return g.subgraph(seed_nodes)

The following modification will tell the dataloader to prefetch the node data named
``feat`` from all the nodes included in ``sg``, and assign it to ``sg.ndata['x']``.

.. code:: python

   class ExampleSubgraphSampler(object):
       def sample(self, g, seed_nodes):
           sg = g.subgraph(seed_nodes)
           sg.ndata['x'] = dgl.LazyFeature('feat')
           return sg
