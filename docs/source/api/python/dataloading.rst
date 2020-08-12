.. _api-sampling:

dgl.dataloading
=================================

.. automodule:: dgl.dataloading

PyTorch node/edge DataLoaders
-----------------------------

.. autoclass:: pytorch.NodeDataLoader
.. autoclass:: pytorch.EdgeDataLoader

General collating functions
---------------------------

.. autoclass:: Collator
.. autoclass:: NodeCollator
.. autoclass:: EdgeCollator

Base Multi-layer Neighborhood Sampling Class
--------------------------------------------

.. autoclass:: BlockSampler

Uniform Node-wise Neighbor Sampling (GraphSAGE style)
-----------------------------------------------------

.. autoclass:: MultiLayerNeighborSampler

Negative Samplers for Link Prediction
-------------------------------------

.. autoclass:: negative_sampler.Uniform
