.. _guide-data-pipeline:

Chapter 4: Graph Data Pipeline
====================================================

DGL implements many commonly used graph datasets in :ref:`apidata`. They
follow a standard pipeline defined in class :class:`dgl.data.DGLDataset`. We highly
recommend processing graph data into a :class:`dgl.data.DGLDataset` subclass, as the
pipeline provides simple and clean solution for loading, processing and
saving graph data.

This chapter introduces how to create a DGL-Dataset for our own graph
data. The following contents explain how the pipeline works, and
show how to implement each component of it.

Roadmap
-------

* :ref:`guide-data-pipeline-dataset`
* :ref:`guide-data-pipeline-download`
* :ref:`guide-data-pipeline-process`
* :ref:`guide-data-pipeline-savenload`
* :ref:`guide-data-pipeline-loadogb`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    data-dataset
    data-download
    data-process
    data-savenload
    data-loadogb