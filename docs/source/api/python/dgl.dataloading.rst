.. _api-dataloading:

dgl.dataloading
=================================

.. automodule:: dgl.dataloading

DataLoaders
-----------
.. currentmodule:: dgl.dataloading.pytorch

DGL DataLoader for mini-batch training works similarly to PyTorch's DataLoader.
It has a generator interface that returns mini-batches sampled from some given graphs.
DGL provides two DataLoaders: a ``NodeDataLoader`` for node classification task
and an ``EdgeDataLoader`` for edge/link prediction task.

.. autoclass:: NodeDataLoader
.. autoclass:: EdgeDataLoader

.. _api-dataloading-neighbor-sampling:
Neighbor Sampler
-----------------------------
.. currentmodule:: dgl.dataloading.neighbor

Neighbor samplers are classes that control the behavior of ``DataLoader`` s
to sample neighbors. All of them inherit the base :class:`BlockSampler` class, but implement
different neighbor sampling strategies by overriding the ``sample_frontier`` or
the ``sample_blocks`` methods.

.. autoclass:: BlockSampler
    :members: sample_frontier, sample_blocks

.. autoclass:: MultiLayerNeighborSampler
    :members: sample_frontier
    :show-inheritance:

.. autoclass:: MultiLayerFullNeighborSampler
    :show-inheritance:

.. _api-dataloading-negative-sampling:

Negative Samplers for Link Prediction
-------------------------------------
.. currentmodule:: dgl.dataloading.negative_sampler

Negative samplers are classes that control the behavior of the ``EdgeDataLoader``
to generate negative edges.

.. autoclass:: Uniform
    :members: __call__

Async Copying to/from GPUs
--------------------------
.. currentmodule:: dgl.dataloading

Data can be copied from the CPU to the GPU, or from the GPU to the CPU,
while the GPU is being used for
computation, using the :class:`AsyncTransferer`.

.. autoclass:: AsyncTransferer
    :members: __init__, async_copy

.. autoclass:: async_transferer.Transfer
    :members: wait
