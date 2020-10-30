.. _guide_cn-data-pipeline:

第4章：图数据处理管道
==============================

:ref:`(English Version) <guide-data-pipeline>`

DGL implements many commonly used graph datasets in :ref:`apidata`. They
follow a standard pipeline defined in class :class:`dgl.data.DGLDataset`. DGL highly
recommends processing graph data into a :class:`dgl.data.DGLDataset` subclass, as the
pipeline provides simple and clean solution for loading, processing and
saving graph data.

DGL在 :ref:`apidata` 里实现了很多常用的图数据集。它们遵循了以下类定义的标准管道： :class:`dgl.data.DGLDataset`。
DGL推荐用户将图数据处理为 :class:`dgl.data.DGLDataset` 的子类。该类为导入，处理和保存图数据提供了简单而干净的解决方案。
本章介绍了如何为用户自己的图数据创建一个DGL数据集。以下内容说明了管道的工作方式，并展示了如何实现管道的每个组件。

Roadmap

本章路线图
-----------

This chapter introduces how to create a custom DGL-Dataset.
The following sections explain how the pipeline works, and
shows how to implement each component of it.

这一章介绍了如何定制化DGL的数据集。后续的章节解释了数据管道是如何运作的，并介绍了每个部分是如何实现的。

* :ref:`guide_cn-data-pipeline-dataset`
* :ref:`guide_cn-data-pipeline-download`
* :ref:`guide_cn-data-pipeline-process`
* :ref:`guide_cn-data-pipeline-savenload`
* :ref:`guide_cn-data-pipeline-loadogb`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    data-dataset
    data-download
    data-process
    data-savenload
    data-loadogb