.. _guide_cn-training-graph-classification:

5.4 整图分类
----------------------------------

:ref:`(English Version) <guide-training-graph-classification>`

Instead of a big single graph, sometimes one might have the data in the
form of multiple graphs, for example a list of different types of
communities of people. By characterizing the friendship among people in
the same community by a graph, one can get a list of graphs to classify. In
this scenario, a graph classification model could help identify the type
of the community, i.e. to classify each graph based on the structure and
overall information.

有时用户数据会由多个图组成，而不是单个的大图数据，例如不同类型的人群社区。
通过用图刻画同一社区里人与人间的友谊，可以得到多张用于分类的图。
这个场景里，整图分类模型可以识别社区的类型，即根据结构和整体信息对图分类。

Overview

概述
~~~~~~~~

The major difference between graph classification and node
classification or link prediction is that the prediction result
characterizes the property of the entire input graph. One can perform the
message passing over nodes/edges just like the previous tasks, but also
needs to retrieve a graph-level representation.

整图分类与节点分类或链路预测二者之间的主要区别是，预测结果刻画了整个输入图的属性。
与之前的任务类似，用户在节点或边上进行消息传递不同的是还需要得到整个图的表示。

The graph classification pipeline proceeds as follows:

整图分类的处理流程如下图所示：

.. figure:: https://data.dgl.ai/tutorial/batch/graph_classifier.png
   :alt: Graph Classification Process

   Graph Classification Process
   整图分类过程

From left to right, the common practice is:

-  Prepare a batch of graphs
-  Perform message passing on the batched graphs to update node/edge features
-  Aggregate node/edge features into graph-level representations
-  Classify graphs based on graph-level representations

从左至右，一般流程是：

-  准备一个批次的图；
-  在成批次的图上进行消息传递模型以更新节点或边的特征；
-  将节点或边特征聚合成整张图的图表示；
-  根据任务设计分类层。

Batch of Graphs

批次的图
^^^^^^^^^^^^^^^

Usually a graph classification task trains on a lot of graphs, and it
will be very inefficient to use only one graph at a time when
training the model. Borrowing the idea of mini-batch training from
common deep learning practice, one can build a batch of multiple graphs
and send them together for one training iteration.

整图分类任务通常需要在很多图上进行训练。如果用户在训练模型时一次仅使用一张图将会非常低效。
借由深度学习实践中常用的小批次训练方法，用户可将多张图组成一个批次，在整个图批次上进行一次训练迭代。

In DGL, one can build a single batched graph from a list of graphs. This
batched graph can be simply used as a single large graph, with connected
components corresponding to the original small graphs.

使用DGL，用户可将一系列的图建立成一个图批次。一个图批次可以被看作是一张大图，图中的每个连通子图对应一张原始小图。

.. figure:: https://data.dgl.ai/tutorial/batch/batch.png
   :alt: Batched Graph

   Batched Graph
   批次化的图

Graph Readout

图读出
^^^^^^^^^^^^^

Every graph in the data may have its unique structure, as well as its
node and edge features. In order to make a single prediction, one usually
aggregates and summarizes over the possibly abundant information. This
type of operation is named *readout*. Common readout operations include
summation, average, maximum or minimum over all node or edge features.

数据集中的每一张图都有它独特的结构和节点与边的特征。为了完成一次预测，通常会聚合并汇总尽可能多的信息。
这类操作叫做“读出”。常见的聚合方法包括：对所有节点或边特征的求和、取平均值、求最大值或最小值。

Given a graph :math:`g`, one can define the average node feature readout as

给定一张图 :math:`g`，对它所有节点特征取平均值的聚合如下：

.. math:: h_g = \frac{1}{|\mathcal{V}|}\sum_{v\in \mathcal{V}}h_v

where :math:`h_g` is the representation of :math:`g`, :math:`\mathcal{V}` is
the set of nodes in :math:`g`, :math:`h_v` is the feature of node :math:`v`.

其中，:math:`h_g` 是图 :math:`g` 的表征， :math:`\mathcal{V}` 是图 :math:`g` 中节点的集合，
:math:`h_v` 是节点 :math:`v` 的特征。

DGL provides built-in support for common readout operations. For example,
:func:`dgl.readout_nodes` implements the above readout operation.

DGL内置了常见的图读出函数，例如 :func:`dgl.readout_nodes` 就实现了上述的读出计算。

Once :math:`h_g` is available, one can pass it through an MLP layer for
classification output.

在得到 :math:`h_g` 后，用户可将其传给一个多层感知机(MLP)层来获得分类输出。

Writing Neural Network Model

编写神经网络模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The input to the model is the batched graph with node and edge features.

模型的输入是带节点和边特征的批次化图。需要注意的是批次化图中的节点和边属性没有批次大小对应的维度。
模型中应特别注意以下几点。

Computation on a Batched Graph

批次化图上的计算
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

接下来讨论批次化图计算的特性。

First, different graphs in a batch are entirely separated, i.e. no edges
between any two graphs. With this nice property, all message passing
functions still have the same results.

首先，一个批次中不同的图是完全分开的，即任意两个图之间没有边连接。
根据这个良好的性质，所有消息传递函数仍然具有相同的结果。

Second, the readout function on a batched graph will be conducted over
each graph separately. Assuming the batch size is :math:`B` and the
feature to be aggregated has dimension :math:`D`, the shape of the
readout result will be :math:`(B, D)`.

其次，读出函数会分别作用在图批次中的每张图上。假设批次大小为 :math:`B`，要聚合的特征大小为 :math:`D`，
则图读出的张量形状为 :math:`(B, D)`。

.. code:: python

    import dgl
    import torch

    g1 = dgl.graph(([0, 1], [1, 0]))
    g1.ndata['h'] = torch.tensor([1., 2.])
    g2 = dgl.graph(([0, 1], [1, 2]))
    g2.ndata['h'] = torch.tensor([1., 2., 3.])
    
    dgl.readout_nodes(g1, 'h')
    # tensor([3.])  # 1 + 2
    
    bg = dgl.batch([g1, g2])
    dgl.readout_nodes(bg, 'h')
    # tensor([3., 6.])  # [1 + 2, 1 + 2 + 3]

Finally, each node/edge feature in a batched graph is obtained by
concatenating the corresponding features from all graphs in order.

最后，批次化图中的每个节点或边特征张量均通过将所有图上的相应特征拼接得到。

.. code:: python

    bg.ndata['h']
    # tensor([1., 2., 1., 2., 3.])

Model Definition

模型定义
^^^^^^^^^^^^^^^^

Being aware of the above computation rules, one can define a model as follows.

了解了上述计算规则后，用户可以定义一个非常简单的模型。

.. code:: python

    import dgl.nn.pytorch as dglnn
    import torch.nn as nn

    class Classifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes):
            super(Classifier, self).__init__()
            self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
            self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g, h):
            # Apply graph convolution and activation.
            # 应用图卷积和激活函数
            h = F.relu(self.conv1(g, h))
            h = F.relu(self.conv2(g, h))
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                # 使用平均读出计算图表示
                hg = dgl.mean_nodes(g, 'h')
                return self.classify(hg)

Training Loop

训练循环
~~~~~~~~~~~~~

Data Loading

数据加载
^^^^^^^^^^^^

Once the model is defined, one can start training. Since graph
classification deals with lots of relatively small graphs instead of a big
single one, one can train efficiently on stochastic mini-batches
of graphs, without the need to design sophisticated graph sampling
algorithms.

模型定义完成后，用户就可以开始训练模型。由于整图分类处理的是很多相对较小的图，而不是一个大图，
因此通常可以在随机抽取的小批次图上进行高效的训练，而无需设计复杂的图采样算法。

Assuming that one have a graph classification dataset as introduced in
:ref:`guide-data-pipeline`.

以下例子中使用了 :ref:`guide_cn-data-pipeline` 中的整图分类数据集。

.. code:: python

    import dgl.data
    dataset = dgl.data.GINDataset('MUTAG', False)

Each item in the graph classification dataset is a pair of a graph and
its label. One can speed up the data loading process by taking advantage
of the DataLoader, by customizing the collate function to batch the
graphs:

整图分类数据集里的每个数据点是一个图和它标签的对子。为提升数据加载速度，
用户可以在DataLoader里自定义collate函数。

.. code:: python

    def collate(samples):
        graphs, labels = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)
        batched_labels = torch.tensor(labels)
        return batched_graph, batched_labels

Then one can create a DataLoader that iterates over the dataset of
graphs in mini-batches.

随后用户可以创建一个以小批次遍历整个图数据集的DataLoader。

.. code:: python

    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1024,
        collate_fn=collate,
        drop_last=False,
        shuffle=True)

Loop

循环
^^^^

Training loop then simply involves iterating over the dataloader and
updating the model.

训练循环仅涉及遍历dataloader和更新模型参数。

.. code:: python

    import torch.nn.functional as F

    # Only an example, 7 is the input feature size
    model = Classifier(7, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['attr'].float()
            logits = model(batched_graph, feats)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()

For an end-to-end example of graph classification, see
`DGL's GIN example <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__. 
The training loop is inside the
function ``train`` in
`main.py <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/main.py>`__.
The model implementation is inside
`gin.py <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/gin.py>`__
with more components such as using
:class:`dgl.nn.pytorch.GINConv` (also available in MXNet and Tensorflow)
as the graph convolution layer, batch normalization, etc.

DGL实现了一份使用图同构网络作整图分类的范例：
`DGL的GIN样例 <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gin>`__。
训练循环的代码请参考位于
`main.py <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/main.py>`__ 源文件中的 ``train`` 函数。
图同构网络的实现位于
`gin.py <https://github.com/dmlc/dgl/blob/master/examples/pytorch/gin/gin.py>`__ ，
其中使用了更多的模块组件，例如使用 :class:`dgl.nn.pytorch.GINConv` 模块作为图卷积层(DGL同样支持MXNet和TensorFlow后端)、批量归一化等。

Heterogeneous graph

异构图上的训练循环
~~~~~~~~~~~~~~~~~~~

Graph classification with heterogeneous graphs is a little different
from that with homogeneous graphs. In addition to graph convolution modules
compatible with heterogeneous graphs, one also needs to aggregate over the nodes of
different types in the readout function.

在异构图上做整图分类和在同构图上做整图分类略有不同。用户除了需要使用异构图卷积模块，还需要在读出函数中聚合不同类别的节点。

The following shows an example of summing up the average of node
representations for each node type.

以下代码示范了对每种节点类型的节点表示平均值求和。

.. code:: python

    class RGCN(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats, rel_names):
            super().__init__()
    
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
    
        def forward(self, graph, inputs):
            # inputs is features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h
    
    class HeteroClassifier(nn.Module):
        def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
            super().__init__()

            self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
            self.classify = nn.Linear(hidden_dim, n_classes)
    
        def forward(self, g):
            h = g.ndata['feat']
            h = self.rgcn(g, h)
            with g.local_scope():
                g.ndata['h'] = h
                # Calculate graph representation by average readout.
                hg = 0
                for ntype in g.ntypes:
                    hg = hg + dgl.mean_nodes(g, 'h', ntype=ntype)
                return self.classify(hg)

The rest of the code is not different from that for homogeneous graphs.

剩余部分的训练代码和同构图代码相同。

.. code:: python

    # etypes is the list of edge types as strings.
    model = HeteroClassifier(10, 20, 5, etypes)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            logits = model(batched_graph)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
