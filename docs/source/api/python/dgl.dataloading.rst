.. _api-dataloading:

dgl.dataloading
=================================

.. automodule:: dgl.dataloading
.. currentmodule:: dgl.dataloading.pytorch

DataLoaders
-----------

DGL DataLoader for mini-batch training works similarly to PyTorch's DataLoader.
It has a generator interface that returns mini-batches sampled from some given graphs.
DGL provides two DataLoaders: a ``NodeDataLoader`` for node classification task
and an ``EdgeDataLoader`` for edge/link prediction task.

.. autoclass:: NodeDataLoader
.. autoclass:: EdgeDataLoader

.. _api-dataloading-neighbor-sampling:

Neighbor Sampler
-----------------------------

Neighbor samplers are classes that control the behavior of ``DataLoader`` s
to sample neighbors. All of them inherit the base :class:`BlockSampler` class, but implement
different neighbor sampling strategies by overriding the ``sample_frontier`` or
the ``sample_blocks`` methods.

.. autoclass:: BlockSampler
    :members: sample_frontier, sample_blocks

.. currentmodule:: dgl.dataloading.neighbor

.. autoclass:: MultiLayerNeighborSampler
    :members: sample_frontier
    :show-inheritance:

.. autoclass:: MultiLayerFullNeighborSampler
    :show-inheritance:

.. _api-dataloading-negative-sampling:

Negative Samplers for Link Prediction
-------------------------------------

Negative samplers are classes that control the behavior of the ``EdgeDataLoader``
to generate negative edges.

.. currentmodule:: dgl.dataloading.negative_sampler

.. autoclass:: Uniform
    :members: __call__

Collators
---------------------------

Collators are adaptor classes to make DGL's neighbor samplers or negative samplers
compatible with the collate function in PyTorch's DataLoader. In most cases, the
``NodeDataLoader`` and ``EdgeDataLoader`` are sufficient to train a GNN stochastically.
The collators are useful when the users need more flexible control or wish to implement
new sampling algorithms that go out of the scope of the ``BlockSampler`` framework.

.. currentmodule:: dgl.dataloading.dataloader

.. autoclass:: Collator
    :members: dataset, collate

.. autoclass:: NodeCollator
    :members: dataset, collate
    :show-inheritance:

.. autoclass:: EdgeCollator
    :members: dataset, collate
    :show-inheritance:
