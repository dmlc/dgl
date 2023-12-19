.. _stochastic_training-ondisk-dataset:

Creating OnDiskDataset
======================

This tutorial shows how to create an `OnDiskDataset` from raw data and use it
for stochastic training.

**GraphBolt** provides the ``OnDiskDataset`` class to help user organize plain
data of graph strucutre, feature data and tasks. ``OnDiskDataset`` is also
designed to efficiently handle large graphs and features that do not fit into
memory by storing them on disk.

For more details about `OnDiskDataset`, please refer to the
:class:`~dgl.graphbolt.OnDiskDataset` API documentation.

.. toctree::
    :maxdepth: 1
    :glob:

    ondisk_dataset_homograph.nblink
    ondisk_dataset_heterograph.nblink
    ondisk-dataset-specification.rst
