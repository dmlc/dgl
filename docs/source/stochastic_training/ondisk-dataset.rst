.. _stochastic_training-ondisk-dataset:

Composing OnDiskDataset from raw data
=====================================

This tutorial shows how to compose :class:`~dgl.graphbolt.OnDiskDataset` from
raw data. A full specification of ``metadata.yaml`` is also provided.

**GraphBolt** provides the ``OnDiskDataset`` class to help user organize plain
data of graph strucutre, feature data and tasks. ``OnDiskDataset`` is also
designed to efficiently handle large graphs and features that do not fit into
memory by storing them on disk.

.. toctree::
    :maxdepth: 1
    :glob:

    ondisk_dataset_homograph.nblink
    ondisk_dataset_heterograph.nblink
    ondisk-dataset-specification.rst
