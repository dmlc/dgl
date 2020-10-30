.. _guide_cn-data-pipeline-dataset:

4.1 DGLDataset类
--------------------

:ref:`(English Version) <guide-data-pipeline-dataset>`

:class:`~dgl.data.DGLDataset` is the base class for processing, loading and saving
graph datasets defined in :ref:`apidata`. It implements the basic pipeline
for processing graph data. The following flow chart shows how the
pipeline works.

:class:`~dgl.data.DGLDataset` 是处理，导入和保存 :ref:`apidata` 中定义的图数据集的基类。
它实现了用于处理图数据的基本管道。以下流程图展示了管道的工作方式。

To process a graph dataset located in a remote server or local disk, one can
define a class, say ``MyDataset``, inheriting from :class:`dgl.data.DGLDataset`. The
template of ``MyDataset`` is as follows.

为了处理位于远程服务器或本地磁盘上的图数据集，我们定义了一个类，称为 ``MyDataset``, 它继承自 :class:`dgl.data.DGLDataset`。
``MyDataset`` 的模版如下图所示。

.. figure:: https://data.dgl.ai/asset/image/userguide_data_flow.png
    :align: center

    Flow chart for graph data input pipeline defined in class DGLDataset.

    在类DGLDataset中定义的图数据输入管道的流程图。

.. code:: 

    from dgl.data import DGLDataset
    
    class MyDataset(DGLDataset):
        """ Template for customizing graph datasets in DGL.
        """ 用于在DGL中自定义图数据集的模板：
    
        Parameters
        ----------
        url : str
            URL to download the raw dataset
            下载原始数据集的url。
        raw_dir : str
            Specifying the directory that will store the 
            downloaded data or the directory that
            already stores the input data.
            Default: ~/.dgl/
            指定将存储下载数据的目录或已存储输入数据的目录。默认: ~/.dgl/
        save_dir : str
            Directory to save the processed dataset.
            Default: the value of `raw_dir`
            处理完成的数据集的保存目录。默认：raw_dir
        force_reload : bool
            Whether to reload the dataset. Default: False
            是否重新导入数据集。默认：False
        verbose : bool
            Whether to print out progress information
            是否打印进度信息。
        """
        def __init__(self, 
                     url=None, 
                     raw_dir=None, 
                     save_dir=None, 
                     force_reload=False, 
                     verbose=False):
            super(MyDataset, self).__init__(name='dataset_name',
                                            url=url,
                                            raw_dir=raw_dir,
                                            save_dir=save_dir,
                                            force_reload=force_reload,
                                            verbose=verbose)
    
        def download(self):
            # download raw data to local disk
            # 将原始数据下载到本地磁盘
            pass
    
        def process(self):
            # process raw data to graphs, labels, splitting masks
            # 将原始数据处理为图，标签，数据集划分掩码
            pass
        
        def __getitem__(self, idx):
            # get one example by index
            # 通过idx得到与之对应的一个样本
            pass
    
        def __len__(self):
            # number of data examples
            # 数据样本的数量
            pass
    
        def save(self):
            # save processed data to directory `self.save_path`
            # 将处理后的数据保存至 `self.save_path`
            pass
    
        def load(self):
            # load processed data from directory `self.save_path`
            # 从 `self.save_path` 导入处理后的数据
            pass
    
        def has_cache(self):
            # check whether there are processed data in `self.save_path`
            # 在 `self.save_path` 中检查是否有处理后的数据
            pass


:class:`~dgl.data.DGLDataset` class has abstract functions ``process()``,
``__getitem__(idx)`` and ``__len__()`` that must be implemented in the
subclass. DGL also recommends implementing saving and loading as well,
since they can save significant time for processing large datasets, and
there are several APIs making it easy (see :ref:`guide-data-pipeline-savenload`).

:class:`~dgl.data.DGLDataset` 类有抽象函数 ``process()``，
``__getitem__(idx)`` 和 ``__len__()``。子类必须实现这些函数。同时DGL也建议实现保存和导入函数，
因为对于处理大型数据集它们可以节省大量时间，并且有多个API可以简化此操作（请参阅 :ref:`guide_cn-data-pipeline-savenload`）。

Note that the purpose of :class:`~dgl.data.DGLDataset` is to provide a standard and
convenient way to load graph data. One can store graphs, features,
labels, masks and basic information about the dataset, such as number of
classes, number of labels, etc. Operations such as sampling, partition
or feature normalization are done outside of the :class:`~dgl.data.DGLDataset`
subclass.

请注意， :class:`~dgl.data.DGLDataset` 的目的是提供一种标准且方便的方式来导入图数据。
用户可以存储有关数据集的图、特征、标签、掩码和例如类别数、标签数等基本信息。
例如采样、划分或特征归一化等操作在 :class:`~dgl.data.DGLDataset` 子类外完成。

The rest of this chapter shows the best practices to implement the
functions in the pipeline.

本章的后续部分展示了在管道中实现这些函数的最佳实践。