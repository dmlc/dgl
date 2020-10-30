.. _guide_cn-data-pipeline-process:

4.3 处理数据
----------------

:ref:`(English Version) <guide-data-pipeline-process>`

One can implement the data processing code in function ``process()``, and it
assumes that the raw data is located in ``self.raw_dir`` already. There
are typically three types of tasks in machine learning on graphs: graph
classification, node classification, and link prediction. This section will show
how to process datasets related to these tasks.

用户可以在 ``process()`` 函数中实现数据处理代码。该函数假定原始数据已经位于 ``self.raw_dir`` 目录中。
图上的机器学习任务通常有三种类型：整图分类，节点分类和链路预测。本节将展示如何处理与这些任务相关的数据集。

The section focuses on the standard way to process graphs, features and masks.
It will use builtin datasets as examples and skip the implementations
for building graphs from files, but add links to the detailed
implementations. Please refer to :ref:`guide-graph-external` to see a
complete guide on how to build graphs from external sources.

展示内容重点介绍处理图、特征和掩码的标准方法。指南将以内置数据集为例，并跳过从文件构建图的实现，
但提供了指向详细实现的链接。请参考 :ref:`guide_cn-graph-external` 以查看有关如何从外部源构建图的完整指南。

Processing Graph Classification datasets

处理整图分类数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Graph classification datasets are almost the same as most datasets in
typical machine learning tasks, where mini-batch training is used. So one can
process the raw data to a list of :class:`dgl.DGLGraph` objects and a list of
label tensors. In addition, if the raw data has been split into
several files, one can add a parameter ``split`` to load specific part of
the data.

整图分类数据集与用于小批次训练的典型机器学习任务中的大多数数据集类似。
因此，我们将原始数据处理为 :class:`dgl.DGLGraph` 对象的列表和标签张量的列表。
此外，如果原始数据已拆分为多个文件，则可以添加参数 ``split`` 以导入数据的特定部分。

Take :class:`~dgl.data.QM7bDataset` as example:

以 :class:`~dgl.data.QM7bDataset` 为例：

.. code::

    from dgl.data import DGLDataset

    class QM7bDataset(DGLDataset):
        _url = 'http://deepchem.io.s3-website-us-west-1.amazonaws.com/' \
               'datasets/qm7b.mat'
        _sha1_str = '4102c744bb9d6fd7b40ac67a300e49cd87e28392'
    
        def __init__(self, raw_dir=None, force_reload=False, verbose=False):
            super(QM7bDataset, self).__init__(name='qm7b',
                                              url=self._url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)
    
        def process(self):
            mat_path = self.raw_path + '.mat'
            # process data to a list of graphs and a list of labels
            # 将数据处理为图列表和标签列表
            self.graphs, self.label = self._load_graph(mat_path)
        
        def __getitem__(self, idx):
            """ Get graph and label by index
            通过idx获取对应的图和标签
    
            Parameters
            ----------
            idx : int
                Item index
    
            Returns
            -------
            (dgl.DGLGraph, Tensor)
            """
            return self.graphs[idx], self.label[idx]
    
        def __len__(self):
            """Number of graphs in the dataset"""
            """数据集中图的数量"""
            return len(self.graphs)


In ``process()``, the raw data is processed to a list of graphs and a
list of labels. One must implement ``__getitem__(idx)`` and ``__len__()``
for iteration. DGL recommends making ``__getitem__(idx)`` return a
tuple ``(graph, label)`` as above. Please check the `QM7bDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/qm7b.html#QM7bDataset>`__
for details of ``self._load_graph()`` and ``__getitem__``.

函数 ``process()`` 将原始数据处理为图列表和标签列表。用户必须实现 ``__getitem__(idx)`` 和  ``__len__()`` 以进行迭代。
DGL建议使 ``__getitem__(idx)`` 返回上面的元组 ``(图，标签)``。
请参照 `QM7bDataset源代码  <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/qm7b.html#QM7bDataset>`__
以获取 ``self._load_graph()`` 和 ``__getitem__`` 的详细信息。

One can also add properties to the class to indicate some useful
information of the dataset. In :class:`~dgl.data.QM7bDataset`, one can add a property
``num_labels`` to indicate the total number of prediction tasks in this
multi-task dataset:

用户还可以向类添加属性以指示一些有用的数据集信息。在 :class:`~dgl.data.QM7bDataset` 中，
用户可以添加属性 ``num_labels`` 来指示此多任务数据集中的预测任务总数：

.. code::

    @property
    def num_labels(self):
        """Number of labels for each graph, i.e. number of prediction tasks."""
        """每个图的标签数，即预测任务数。"""
        return 14

After all these coding, one can finally use :class:`~dgl.data.QM7bDataset` as
follows:

在编写完这些代码之后，最终可以如下使用 :class:`~dgl.data.QM7bDataset`：

.. code:: 

    import dgl
    import torch

    from torch.utils.data import DataLoader
    
    # load data
    # 数据导入
    dataset = QM7bDataset()
    num_labels = dataset.num_labels
    
    # create collate_fn
    # 创建 collate_fn
    def _collate_fn(batch):
        graphs, labels = batch
        g = dgl.batch(graphs)
        labels = torch.tensor(labels, dtype=torch.long)
        return g, labels
    
    # create dataloaders
    # 创建 dataloaders
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=_collate_fn)
    
    # training
    # 训练
    for epoch in range(100):
        for g, labels in dataloader:
            # your training code here
            # 用户的训练代码
            pass

A complete guide for training graph classification models can be found
in :ref:`guide-training-graph-classification`.

训练整图分类模型的完整指南可以在5.4整图分类中找到。

For more examples of graph classification datasets, please refer to DGL's builtin graph classification
datasets: 

有关整图分类数据集的更多示例，请参考 :ref:`guide-training-graph-classification`：

* :ref:`gindataset`

* :ref:`minigcdataset`

* :ref:`qm7bdata`

* :ref:`tudata`

Processing Node Classification datasets

处理节点分类数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Different from graph classification, node classification is typically on
a single graph. As such, splits of the dataset are on the nodes of the
graph. DGL recommends using node masks to specify the splits. The section uses
builtin dataset `CitationGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__ as an example:

与整图分类不同，节点分类通常在单个图上进行。因此，数据集的划分在图的节点上进行。
DGL建议使用节点掩码来指定拆分。以内置数据集 `CitationGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__ 为例：

.. code::

    from dgl.data import DGLBuiltinDataset
    from dgl.data.utils import _get_dgl_url, generate_mask_tensor
    
    class CitationGraphDataset(DGLBuiltinDataset):
        _urls = {
            'cora_v2' : 'dataset/cora_v2.zip',
            'citeseer' : 'dataset/citeseer.zip',
            'pubmed' : 'dataset/pubmed.zip',
        }
    
        def __init__(self, name, raw_dir=None, force_reload=False, verbose=True):
            assert name.lower() in ['cora', 'citeseer', 'pubmed']
            if name.lower() == 'cora':
                name = 'cora_v2'
            url = _get_dgl_url(self._urls[name])
            super(CitationGraphDataset, self).__init__(name,
                                                       url=url,
                                                       raw_dir=raw_dir,
                                                       force_reload=force_reload,
                                                       verbose=verbose)
    
        def process(self):
            # Skip some processing code
            # === data processing skipped ===
    
            # build graph
            # 跳过一些处理的代码
            # === 跳过数据处理 ===
            # 构建图
            g = dgl.graph(graph)
            # splitting masks
            # 划分掩码
            g.ndata['train_mask'] = generate_mask_tensor(train_mask)
            g.ndata['val_mask'] = generate_mask_tensor(val_mask)
            g.ndata['test_mask'] = generate_mask_tensor(test_mask)
            # node labels
            # 节点标签
            g.ndata['label'] = torch.tensor(labels)
            # node features
            # 节点特征
            g.ndata['feat'] = torch.tensor(_preprocess_features(features),
                                           dtype=F.data_type_dict['float32'])
            self._num_labels = onehot_labels.shape[1]
            self._labels = labels
            self._g = g
    
        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g
    
        def __len__(self):
            return 1

For brevity, this section skips some code in ``process()`` to highlight the key
part for processing node classification dataset: splitting masks. Node
features and node labels are stored in ``g.ndata``. For detailed
implementation, please refer to `CitationGraphDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__.

为简便起见，这里省略了 ``process()`` 中的一些代码，以突出展示用于处理节点分类数据集的关键部分：
分割掩码。节点特征和节点标签被存储在 ``g.ndata`` 中。详细的实现请参考
`CitationGraphDataset源代码 <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/citation_graph.html#CitationGraphDataset>`__ 。

Note that the implementations of ``__getitem__(idx)`` and
``__len__()`` are changed as well, since there is often only one graph
for node classification tasks. The masks are ``bool tensors`` in PyTorch
and TensorFlow, and ``float tensors`` in MXNet.

注意， ``__getitem__(idx)`` 和 ``__len__()`` 的实现也发生了变化。
这是因为节点分类任务通常只用一个图。掩码在PyTorch和TensorFlow中是bool张量，在MXNet中是float张量。

The section uses a subclass of ``CitationGraphDataset``, :class:`dgl.data.CiteseerGraphDataset`,
to show the usage of it:

本节中使用 :class:`dgl.data.CiteseerGraphDataset` 的子类 ``CitationGraphDataset`` 来展示其用法：

.. code:: 

    # load data
    # 导入数据
    dataset = CiteseerGraphDataset(raw_dir='')
    graph = dataset[0]
    
    # get split masks
    # 获取划分的掩码
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    
    # get node features
    # 获取节点特征
    feats = graph.ndata['feat']
    
    # get labels
    # 获取标签
    labels = graph.ndata['label']

A complete guide for training node classification models can be found in
:ref:`guide-training-node-classification`.

:ref:`guide-training-node-classification` 提供了训练节点分类模型的完整指南。

For more examples of node classification datasets, please refer to DGL's
builtin datasets:

有关节点分类数据集的更多示例，请参阅以下内置数据集：

* :ref:`citationdata`

* :ref:`corafulldata`

* :ref:`amazoncobuydata`

* :ref:`coauthordata`

* :ref:`karateclubdata`

* :ref:`ppidata`

* :ref:`redditdata`

* :ref:`sbmdata`

* :ref:`sstdata`

* :ref:`rdfdata`

Processing dataset for Link Prediction datasets

处理链路预测数据集
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The processing of link prediction datasets is similar to that for node
classification’s, there is often one graph in the dataset.

链路预测数据集的处理与节点分类的处理相似，数据集中通常只有一个图。

The section uses builtin dataset
`KnowledgeGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
as an example, and still skips the detailed data processing code to
highlight the key part for processing link prediction datasets:

本节以内置的数据集 `KnowledgeGraphDataset <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
为例，同时省略了详细的数据处理代码以突出展示处理链路预测数据集的关键部分：

.. code::

    # Example for creating Link Prediction datasets
    # 创建链路预测数据集示例
    class KnowledgeGraphDataset(DGLBuiltinDataset):
        def __init__(self, name, reverse=True, raw_dir=None, force_reload=False, verbose=True):
            self._name = name
            self.reverse = reverse
            url = _get_dgl_url('dataset/') + '{}.tgz'.format(name)
            super(KnowledgeGraphDataset, self).__init__(name,
                                                        url=url,
                                                        raw_dir=raw_dir,
                                                        force_reload=force_reload,
                                                        verbose=verbose)
    
        def process(self):
            # Skip some processing code
            # === data processing skipped ===
    
            # splitting mask
            # 跳过一些处理的代码
            # === 跳过数据处理 ===
            # 划分掩码
            g.edata['train_mask'] = train_mask
            g.edata['val_mask'] = val_mask
            g.edata['test_mask'] = test_mask
            # edge type
            # 边类型
            g.edata['etype'] = etype
            # node type
            # 节点类型
            g.ndata['ntype'] = ntype
            self._g = g
    
        def __getitem__(self, idx):
            assert idx == 0, "This dataset has only one graph"
            return self._g
    
        def __len__(self):
            return 1

As shown in the code, it adds splitting masks into ``edata`` field of the
graph. Check `KnowledgeGraphDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
to see the complete code. The following code uses a subclass of ``KnowledgeGraphDataset``,
:class:`dgl.data.FB15k237Dataset`, to show the usage of it:

如代码所示，图的 ``edata`` 存储了划分掩码。在
`KnowledgeGraphDataset source
code <https://docs.dgl.ai/en/0.5.x/_modules/dgl/data/knowledge_graph.html#KnowledgeGraphDataset>`__
源代码中可以查看完整的代码。指南使用 ``KnowledgeGraphDataset``的子类 :class:`dgl.data.FB15k237Dataset` 来展示其用法：

.. code:: 

    from dgl.data import FB15k237Dataset

    # load data
    # 导入数据
    dataset = FB15k237Dataset()
    graph = dataset[0]
    
    # get training mask
    # 获取训练集掩码
    train_mask = graph.edata['train_mask']
    train_idx = torch.nonzero(train_mask).squeeze()
    src, dst = graph.edges(train_idx)
    # get edge types in training set
    # 获取训练集中的边类型
    rel = graph.edata['etype'][train_idx]


A complete guide for training link prediction models can be found in
:ref:`guide-training-link-prediction`.

For more examples of link prediction datasets, please refer to DGL's
builtin datasets: 

有关训练链路预测模型的完整指南，请参见:ref:`guide-training-link-prediction`。

有关链路预测数据集的更多示例，请参考DGL的内置数据集：

* :ref:`kgdata`

* :ref:`bitcoinotcdata`
