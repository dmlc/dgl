.. _guide-minibatch-gpu-sampling:

6.7 Using GPU for Neighborhood Sampling
---------------------------------------

DGL since 0.7 has been supporting GPU-based neighborhood sampling, which we discuss in this section.

.. note::

   This feature is experimental and a work-in-progress.  Please stay tuned for further
   updates.

Using GPU-based neighborhood sampling in DGL data loaders
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If one is using both

* DGL's builtin data loader (i.e. :class:`~dgl.dataloading.pytorch.NodeDataLoader`), and

* :ref:`DGL's builtin neighborhood sampler <api-dataloading-neighbor-sampling>`, or
  neighborhood samplers subclassed from :class:`~dgl.dataloading.neighbor.BlockSampler`.

Then DGL data loaders will select the device to sample neighbors and construct MFGs
depending on the device of the input graph:

* If the input graph is on GPU, the argument :attr:`device` must be on GPU and
  :attr:`num_workers` must be 0.  In this case, both neighbor sampling and conversion
  to MFG happens on GPU.  This is best for when one can fit the graph onto GPU.

* If the input graph is on CPU, device selection depends on the argument :attr:`device`
  and :attr:`num_workers`:

  * If :attr:`device` is GPU and :attr:`num_workers` is 0, neighbor sampling will be
    performed on CPU, but conversion to MFG happens on GPU.

  * If :attr:`device` is GPU but :attr:`num_workers` is not 0, both neighbor sampling
    and conversion to MFG happens on CPU.  The resulting MFGs are later copied to GPU.

.. note::

   Currently :class:`~dgl.dataloading.pytorch.EdgeDataLoader` and heterogeneous graphs
   are not supported.

Using GPU-based neighbor sampling with DGL functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following sampling functions support operating on GPU:

* :func:`dgl.sampling.sample_neighbors`

  * Only has support for uniform sampling; non-uniform sampling can only run on CPU.

Besides the functions above, :func:`dgl.to_block` can also run on GPU.
