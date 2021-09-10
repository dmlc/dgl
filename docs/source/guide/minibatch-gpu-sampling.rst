.. _guide-minibatch-gpu-sampling:

6.7 Using GPU for Neighborhood Sampling
---------------------------------------

DGL since 0.7 has been supporting GPU-based neighborhood sampling, which has a significant
speed advantage over CPU-based neighborhood sampling.  If you estimate that your graph and
its features can fit onto GPU and your model does not take a lot of GPU memory, then it is
best to put the GPU into memory and use GPU-based neighbor sampling.

For example, `OGB Products <https://ogb.stanford.edu/docs/nodeprop/#ogbn-products>`_ has
2.4M nodes and 61M edges, each node having 100-dimensional features.  The node feature
themselves take less than 1GB memory, and the graph also takes less than 1GB since the
memory consumption of a graph depends on the number of edges.  Therefore it is entirely
possible to fit the whole graph onto GPU.

.. note::

   This feature is experimental and a work-in-progress.  Please stay tuned for further
   updates.

Using GPU-based neighborhood sampling in DGL data loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can use GPU-based neighborhood sampling with DGL data loaders via

* Putting the graph onto GPU.

* Set ``num_workers`` argument to 0, because CUDA does not allow multiple processes
  accessing the same context.
  
* Set ``device`` argument to a GPU device.

All the other arguments for the :class:`~dgl.dataloading.pytorch.NodeDataLoader` can be
the same as the other user guides and tutorials.

.. code:: python

   g = g.to('cuda:0')
   dataloader = dgl.dataloading.NodeDataLoader(
       g,                                # The graph must be on GPU.
       train_nid,
       sampler,
       device=torch.device('cuda:0'),    # The device argument must be GPU.
       num_workers=0,                    # Number of workers must be 0.
       batch_size=1000,
       drop_last=False,
       shuffle=True)
       
GPU-based neighbor sampling also works for custom neighborhood samplers as long as
(1) your sampler is subclassed from :class:`~dgl.dataloading.BlockSampler`, and (2)
your sampler entirely works on GPU.

.. note::

   Currently :class:`~dgl.dataloading.pytorch.EdgeDataLoader` and heterogeneous graphs
   are not supported.

Using GPU-based neighbor sampling with DGL functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following sampling functions support operating on GPU:

* :func:`dgl.sampling.sample_neighbors`

  * Only has support for uniform sampling; non-uniform sampling can only run on CPU.

Besides the functions above, :func:`dgl.to_block` can also run on GPU.
