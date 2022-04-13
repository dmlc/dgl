.. _guide-data-pipeline:

Chapter 4: Graph Data Pipeline
==============================

:ref:`(中文版) <guide_cn-data-pipeline>`

DGL implements many commonly used graph datasets in :ref:`apidata`. They
follow a standard pipeline defined in class :class:`dgl.data.DGLDataset`. DGL highly
recommends processing graph data into a :class:`dgl.data.DGLDataset` subclass, as the
pipeline provides simple and clean solution for loading, processing and
saving graph data.

Roadmap
-------

This chapter introduces how to create a custom DGL-Dataset.
The following sections explain how the pipeline works, and
shows how to implement each component of it.

* :ref:`guide-data-pipeline-dataset`
* :ref:`guide-data-pipeline-download`
* :ref:`guide-data-pipeline-process`
* :ref:`guide-data-pipeline-savenload`
* :ref:`guide-data-pipeline-loadogb`
* :ref:`guide-data-pipeline-loadcsv`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    data-dataset
    data-download
    data-process
    data-savenload
    data-loadogb
    data-loadcsv