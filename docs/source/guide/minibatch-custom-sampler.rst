.. _guide-minibatch-customizing-neighborhood-sampler:

6.4 Implementing custom samplers
----------------------------------------------

Implementing custom samplers involves subclassing the :class:`dgl.dataloading.Sampler`
base class and implementing its abstract :attr:`sample` method.  The :attr:`sample`
method should take in two arguments:

.. code:: python

   def sample(self, g, indices):
       pass

The first argument :attr:`g` is the original graph and the second argument
:attr:`indices` is the indices yielded by the data loader.  Recall that the dataloader
signature is:

.. code:: python

   def __init__(self, graph, indices, graph_sampler, device='cpu', use_ddp=False,
                ddp_seed=0, batch_size=1, drop_last=False, shuffle=False,
                use_prefetch_thread=None, use_alternate_streams=None,
                pin_prefetcher=None, use_uva=False, **kwargs):
       pass

The :attr:`indices` argument will have the same type as the indices argument of the data
loader, i.e. a tensor if the latter is a tensor, and a dictionary of tensors if the
latter is a dictionary of tensors.

An example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The main logic of DGL's neighborhood sampler looks like this:

.. code:: python

   class NeighborSampler(Sampler):
       def __init__(self, fanouts):
           super().__init__()
           self.fanouts = fanouts

       def sample(self, g, seed_nodes):
           output_nodes = seed_nodes
           blocks = []
           for fanout in reversed(self.fanouts):
               # Samples a fixed number of neighbors for the current layer of nodes,
               # inducing a subgraph in the process.
               frontier = g.sample_neighbors(seed_nodes, fanout)
               # Convert this subgraph to a message flow graph (see the stochastic
               # training tutorials for definition).
               block = to_block(frontier, seed_nodes)
               seed_nodes = block.srcdata[NID]
               blocks.insert(0, block)
   
           return seed_nodes, output_nodes, blocks

For example, if you wish to re-implement your own neighbor sampler, but instead
of sampling a fixed number of neighbors, you would like to drop the incoming edges
with a probability, you can replace the statement with ``g.sample_neighbors`` above
with statements that performs edge dropping:

.. code:: python

   class DropoutSampler(Sampler):
       def __init__(self, fanouts):
           super().__init__()
           self.fanouts = fanouts

       def sample(self, g, seed_nodes):
           output_nodes = seed_nodes
           blocks = []
           for fanout in reversed(self.fanouts):
               # Get all the edge IDs.
               sg = g.in_subgraph(seed_nodes)
               # Randomly drop some of them.
               mask = torch.zeros_like(edge_ids).bernoulli_(self.p).bool()
               # Obtain a subgraph induced by those edges.
               frontier = g.edge_subgraph(mask)
               # Convert this subgraph to a message flow graph (see the stochastic
               # training tutorials for definition).
               block = to_block(frontier, seed_nodes)
               seed_nodes = block.srcdata[NID]
               blocks.insert(0, block)
   
           return seed_nodes, output_nodes, blocks

Heterogeneous graphs
~~~~~~~~~~~~~~~~~~~~

If the graph is heterogeneous, the second argument of :attr:`sample` method will be a
dictionary of node IDs.

.. code:: python

   class DropoutSampler(Sampler):
       def __init__(self, fanouts):
           super().__init__()
           self.fanouts = fanouts

       def sample(self, g, seed_nodes):
           output_nodes = seed_nodes
           blocks = []
           for fanout in reversed(self.fanouts):
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
               frontier = dgl.edge_subgraph(new_edges_masks, relabel_nodes=False)
               # Convert this subgraph to a message flow graph (see the stochastic
               # training tutorials for definition).
               block = to_block(frontier, seed_nodes)
               seed_nodes = block.srcdata[NID]
               blocks.insert(0, block)
   
           return seed_nodes, output_nodes, blocks

Implementing custom samplers for use with :func:`dgl.dataloading.as_edge_prediction_sampler`
^^^^^^^^^^

You could wrap your sampler written for node classification into another sampler
for edge classification and link prediction with the same sampling algorithm by calling
:func:`dgl.dataloading.as_edge_prediction_sampler`.  However, sometimes it is better
to exclude the edges related to the ones sampled in the minibatch from neighbor
expansion, as mentioned in :ref:<guide-minibatch-edge-classification-sampler-exclude>.
Therefore, :func:`~dgl.dataloading.as_edge_prediction_sampler` requires the sampler's
:attr:`sample` method to have an additional optional third argument ``exclude_eids``.
An example is given below, adapting the neighbor sampler above to the case where
a given set of edges should be excluded from neighbor expansion:

.. code:: python

   class NeighborSampler(Sampler):
       def __init__(self, fanouts):
           super().__init__()
           self.fanouts = fanouts

       # NOTE: there is an additional third argument
       def sample(self, g, seed_nodes, exclude_eids=None):
           output_nodes = seed_nodes
           blocks = []
           for fanout in reversed(self.fanouts):
               # Samples a fixed number of neighbors for the current layer of nodes,
               # inducing a subgraph in the process.
               frontier = g.sample_neighbors(seed_nodes, fanout, exclude_edges=exclude_eids)
               # Convert this subgraph to a message flow graph (see the stochastic
               # training tutorials for definition).
               block = to_block(frontier, seed_nodes)
               seed_nodes = block.srcdata[NID]
               blocks.insert(0, block)
   
           return seed_nodes, output_nodes, blocks

