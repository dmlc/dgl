.. _guide_cn-training-node-classification:

5.1 节点分类/回归
--------------------------------------------------

:ref:`(English Version) <guide-training-node-classification>`

One of the most popular and widely adopted tasks for graph neural
networks is node classification, where each node in the
training/validation/test set is assigned a ground truth category from a
set of predefined categories. Node regression is similar, where each
node in the training/validation/test set is assigned a ground truth
number.

对于图神经网络来说，最受欢迎和广泛采用的任务之一是节点分类，
其中训练/验证/测试集中的每个节点都从一组预定义的类别中分配一个正确标注的类别。
节点回归也是类似的，其中训练/验证/测试集中的每个节点都被分配了一个正确标注的数字。

Overview

概述
~~~~~~~~

To classify nodes, graph neural network performs message passing
discussed in :ref:`guide-message-passing` to utilize the node’s own
features, but also its neighboring node and edge features. Message
passing can be repeated multiple rounds to incorporate information from
larger range of neighborhood.

为了对节点进行分类，图神经网络执行了 :ref:`guide-message-passing` 中讨论的消息传递来利用节点自身的特征和其邻节点及边的特征。
消息传递可以重复多轮，以纳入更大范围的邻居信息。

Writing neural network model

编写神经网络模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL provides a few built-in graph convolution modules that can perform
one round of message passing. In this guide, we choose
:class:`dgl.nn.pytorch.SAGEConv` (also available in MXNet and Tensorflow),
the graph convolution module for GraphSAGE.

DGL提供了一些内置的图卷积模块，可以进行一轮消息传递。
本指南中选择 :class:`dgl.nn.pytorch.SAGEConv` (在DGL的MXNet和Tensorflow包中也有)，
它是GraphSAGE中使用的图卷积模块。

Usually for deep learning models on graphs we need a multi-layer graph
neural network, where we do multiple rounds of message passing. This can
be achieved by stacking graph convolution modules as follows.

通常对于图上的深度学习模型，需要一个多层图神经网络，并在这个网络中要进行多轮的信息传递。
这可以通过堆叠图卷积模块来实现，具体如下。

.. code:: python

    # Contruct a two-layer GNN model
    # 构建一个2层的GNN模型
    import dgl.nn as dglnn
    import torch.nn as nn
    import torch.nn.functional as F
    class SAGE(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(
                in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
            self.conv2 = dglnn.SAGEConv(
                in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = F.relu(h)
            h = self.conv2(graph, h)
            return h

Note that you can use the model above for not only node classification,
but also obtaining hidden node representations for other downstream
tasks such as
:ref:`guide-training-edge-classification`,
:ref:`guide-training-link-prediction`, or
:ref:`guide-training-graph-classification`.

请注意，这个模型不仅可以做节点分类，还可以为其他下游任务获取隐藏节点表示，如：
:ref:`guide_cn-training-edge-classification`,
:ref:`guide_cn-training-link-prediction`, or
:ref:`guide_cn-training-graph-classification`.


For a complete list of built-in graph convolution modules, please refer
to :ref:`apinn`.

关于DGL内置图卷积模块的完整列表，读者可以参考 :ref:`apinn`。

For more details in how DGL
neural network modules work and how to write a custom neural network
module with message passing please refer to the example in :ref:`guide-nn`.

有关DGL神经网络模块如何工作，以及如何编写一个自定义的带有消息传递的GNN模块的更多细节，请参考 :ref:`guide_cn-nn` 中的例子。

Training loop

训练循环
~~~~~~~~~~~~~

Training on the full graph simply involves a forward propagation of the
model defined above, and computing the loss by comparing the prediction
against ground truth labels on the training nodes.

全图上的训练只需要对上面定义的模型进行正向传播，并通过在训练节点上比较预测和真实标签来计算损失。

This section uses a DGL built-in dataset
:class:`dgl.data.CiteseerGraphDataset` to
show a training loop. The node features
and labels are stored on its graph instance, and the
training-validation-test split are also stored on the graph as boolean
masks. This is similar to what you have seen in :ref:`guide-data-pipeline`.

本节使用DGL内置的数据集 :class:`dgl.data.CiteseerGraphDataset` 来展示一个训练循环。
节点特征和标签存储在其图实例上，训练-验证-测试的分割也以布尔掩码的形式存储在图上。这与在
:ref:`guide_cn-data-pipeline` 中的做法类似。

.. code:: python

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

The following is an example of evaluating your model by accuracy.

下面是一个通过使用准确性来评估模型的例子。

.. code:: python

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

You can then write our training loop as follows.

用户可以按如下方式实现训练循环。

.. code:: python

    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        model.train()
        # forward propagation by using all nodes
        logits = model(graph, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels, valid_mask)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in this example.


`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py>`__
provides an end-to-end homogeneous graph node classification example.
You could see the corresponding model implementation is in the
``GraphSAGE`` class in the example with adjustable number of layers,
dropout probabilities, and customizable aggregation functions and
nonlinearities.

`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py>`__
提供了一个端到端的同构图节点分类的例子。可以在 ``GraphSAGE`` 类中看到对应的模型实现。
这个模型具有可调节的层数、dropout概率，以及可定制的聚合函数和非线性函数。

.. _guide_cn-training-rgcn-node-classification:

Heterogeneous graph

异构图上的训练循环
~~~~~~~~~~~~~~~~~~~

If your graph is heterogeneous, you may want to gather message from
neighbors along all edge types. You can use the module
:class:`dgl.nn.pytorch.HeteroGraphConv` (also available in MXNet and Tensorflow)
to perform message passing
on all edge types, then combining different graph convolution modules
for each edge type.

如果图是异构的，用户可能希望沿着所有边类型从邻居那里收集消息。
用户可以使用模块 :class:`dgl.nn.pytorch.HeteroGraphConv` (也可以在DGL的MXNet和Tensorflow包中使用)在所有边类型上执行消息传递，
然后为每个边类型组合不同的图卷积模块。

The following code will define a heterogeneous graph convolution module
that first performs a separate graph convolution on each edge type, then
sums the message aggregations on each edge type as the final result for
all node types.

下面的代码将定义一个异构图卷积模块，首先对每个边类型进行单独的图卷积，然后将每个边类型上的消息聚合结果相加，作为所有节点类型的最终结果。

.. code:: python

    # Define a Heterograph Conv model
    import dgl.nn as dglnn
    
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
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h

``dgl.nn.HeteroGraphConv`` takes in a dictionary of node types and node
feature tensors as input, and returns another dictionary of node types
and node features.

``dgl.nn.HeteroGraphConv`` 接收一个节点类型和节点特征张量的字典作为输入，并返回另一个节点类型和节点特征的字典。

So given that we have the user and item features in the
:ref:`heterogeneous graph example <guide-training-heterogeneous-graph-example>`.

如下面代码所示，在 :ref:`heterogeneous graph example <guide-training-heterogeneous-graph-example>`
的例子中已经有了用户和项目的特征。

.. code:: python

    model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    labels = hetero_graph.nodes['user'].data['label']
    train_mask = hetero_graph.nodes['user'].data['train_mask']

One can simply perform a forward propagation as follows:

用户可以简单地进行如下的正向传播：

.. code:: python

    node_features = {'user': user_feats, 'item': item_feats}
    h_dict = model(hetero_graph, {'user': user_feats, 'item': item_feats})
    h_user = h_dict['user']
    h_item = h_dict['item']

Training loop is the same as the one for homogeneous graph, except that
now you have a dictionary of node representations from which you compute
the predictions. For instance, if you are only predicting the ``user``
nodes, you can just extract the ``user`` node embeddings from the
returned dictionary:

异构图上的训练循环和同构图的训练循环是一样的，只是现在用户有一个节点表示的字典以计算预测。
例如，如果只预测 ``user`` 节点，用户可以只从返回的字典中提取 ``user`` 的节点嵌入。

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(hetero_graph, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # Compute validation accuracy.  Omitted in this example.
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in the example.


DGL provides an end-to-end example of
`RGCN <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py>`__
for node classification. You can see the definition of heterogeneous
graph convolution in ``RelGraphConvLayer`` in the `model implementation
file <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py>`__.

DGL提供了一个用于节点分类的RGCN的端到端的例子
`RGCN <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py>`__
。用户可以在 `模型实现文件
<https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py>`__
中查看异构图卷积 ``RelGraphConvLayer`` 的定义。


