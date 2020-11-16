.. _guide_cn-data-pipeline:

第4章：图数据处理管道
==============================

:ref:`(English Version) <guide-data-pipeline>`

DGL在 :ref:`apidata` 里实现了很多常用的图数据集。它们遵循了由 :class:`dgl.data.DGLDataset` 类定义的标准的数据处理管道。
DGL推荐用户将图数据处理为 :class:`dgl.data.DGLDataset` 的子类。该类为导入、处理和保存图数据提供了简单而干净的解决方案。

本章路线图
-----------

本章介绍了如何为用户自己的图数据创建一个DGL数据集。以下内容说明了管道的工作方式，并展示了如何实现管道的每个组件。

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