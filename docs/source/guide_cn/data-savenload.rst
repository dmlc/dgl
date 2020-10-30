.. _guide-data-pipeline-savenload:

4.4 保存和加载数据
----------------------

:ref:`(English Version) <guide-data-pipeline-savenload>`

DGL recommends implementing saving and loading functions to cache the
processed data in local disk. This saves a lot of data processing time
in most cases. DGL provides four functions to make things simple:

DGL建议执行保存和加载函数，将处理后的数据缓存在本地磁盘中。
这样在大多数情况下可以节省大量的数据处理时间。DGL提供了4个函数，让事情变得简单。

-  :func:`dgl.save_graphs` and :func:`dgl.load_graphs`: save/load DGLGraph objects and labels to/from local disk.
-  :func:`dgl.data.utils.save_info` and :func:`dgl.data.utils.load_info`: save/load useful information of the dataset (python ``dict`` object) to/from local disk.

-  :func:`dgl.save_graphs` 和 :func:`dgl.load_graphs`: 保存DGLGraph对象和标签到本地磁盘和从本地磁盘读取。
-  :func:`dgl.data.utils.save_info` 和 :func:`dgl.data.utils.load_info`: 将数据集的有用信息保存到本地磁盘和从本地磁盘读取。

The following example shows how to save and load a list of graphs and
dataset information.

下面的示例显示了如何保存和读取图和数据集信息的列表。

.. code:: 

    import os
    from dgl import save_graphs, load_graphs
    from dgl.data.utils import makedirs, save_info, load_info
    
    def save(self):
        # save graphs and labels
        # 保存图和标签
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.graphs, {'labels': self.labels})
        # save other information in python dict
        # 在Python字典里保存其他信息
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        save_info(info_path, {'num_classes': self.num_classes})
    
    def load(self):
        # load processed data from directory `self.save_path`
        # 从目录 `self.save_path` 里读取处理过的数据
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self.graphs, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        self.num_classes = load_info(info_path)['num_classes']
    
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        # 检查在 `self.save_path` 里有处理过的数据文件
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
        return os.path.exists(graph_path) and os.path.exists(info_path)

Note that there are cases not suitable to save processed data. For
example, in the builtin dataset :class:`~dgl.data.GDELTDataset`,
the processed data is quite large, so it’s more effective to process
each data example in ``__getitem__(idx)``.

备注：有些情况下不适合保存处理过的数据。例如，在内置数据集 :class:`~dgl.data.GDELTDataset` 中，
处理过的数据比较大，所以在 ``__getitem__(idx)`` 中处理每个数据实例比较有效。

.. code::

    print(split_edge['valid'].keys())
    print(split_edge['test'].keys())
