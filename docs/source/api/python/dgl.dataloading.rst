.. _api-dataloading:

dgl.dataloading
=================================

.. automodule:: dgl.dataloading

DataLoaders
-----------

PyTorch node/edge DataLoaders
`````````````````````````````

.. currentmodule:: dgl.dataloading.pytorch

.. autoclass:: NodeDataLoader
.. autoclass:: EdgeDataLoader

General collating functions
```````````````````````````

.. currentmodule:: dgl.dataloading.dataloader

.. autoclass:: Collator
    :members: dataset, collate

.. autoclass:: NodeCollator
    :members: dataset, collate
    :show-inheritance:

.. autoclass:: EdgeCollator
    :members: dataset, collate
    :show-inheritance:

.. _api-dataloading-neighbor-sampling:

Neighborhood Sampling Classes
-----------------------------

Base Multi-layer Neighborhood Sampling Class
````````````````````````````````````````````

.. autoclass:: BlockSampler
    :members: sample_frontier, sample_blocks

Uniform Node-wise Neighbor Sampling (GraphSAGE style)
`````````````````````````````````````````````````````

.. currentmodule:: dgl.dataloading.neighbor

.. autoclass:: MultiLayerNeighborSampler
    :members: sample_frontier
    :show-inheritance:

.. autoclass:: MultiLayerFullNeighborSampler
    :show-inheritance:

.. _api-dataloading-negative-sampling:

Negative Samplers for Link Prediction
-------------------------------------

.. currentmodule:: dgl.dataloading.negative_sampler

.. autoclass:: Uniform
    :members: __call__
