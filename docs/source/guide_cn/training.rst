.. _guide_cn-training:


第5章：训练图神经网络
=====================================================

概览
--------

This chapter discusses how to train a graph neural network for node
classification, edge classification, link prediction, and graph
classification for small graph(s), by message passing methods introduced
in :ref:`guide-message-passing` and neural network modules introduced in
:ref:`guide-nn`.

本章通过使用 :ref:`guide_cn-message-passing` 中介绍的消息传递方法和 :ref:`guide-nn` 中介绍的神经网络模块，
讲解了如何对小规模的图进行节点分类、边分类、链接预测和图分类的图神经网络的训练。

This chapter assumes that your graph as well as all of its node and edge
features can fit into GPU; see :ref:`guide-minibatch` if they cannot.

本章假设用户的图以及所有的节点和边特征都能存进GPU；对于无法全部载入的情况，请参见用户指南的 :ref:`guide-minibatch`。

The following text assumes that the graph(s) and node/edge features are
already prepared. If you plan to use the dataset DGL provides or other
compatible ``DGLDataset`` as is described in :ref:`guide-data-pipeline`, you can
get the graph for a single-graph dataset with something like

本章后续的内容均假设用户已经准备好了图和节点及边的特征。如果用户希望使用DGL提供的数据集或其他兼容
 ``DGLDataset`` （如 :ref:`guide_cn-data-pipeline` 所述）的数据，
可以使用类似以下代码的方法获取单个图数据集的图数据。

.. code:: python

    import dgl
    
    dataset = dgl.data.CiteseerGraphDataset()
    graph = dataset[0]


Note: In this chapter we will use PyTorch as backend.

Note: 本章代码使用PyTorch作为DGL的后端框架。

.. _guide-training-heterogeneous-graph-example:

Heterogeneous Graphs

异构图上的图神经网络的训练
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you would like to work on heterogeneous graphs. Here we take a
synthetic heterogeneous graph as an example for demonstrating node
classification, edge classification, and link prediction tasks.

有时用户会想在异构图上进行图神经网络的训练。本章会以下面代码所创建的一个异构图为例，来演示节点分类、边分类和链接预测的训练任务。

The synthetic heterogeneous graph ``hetero_graph`` has these edge types:

这个 ``hetero_graph`` 异构图 有以下这些边的类型：

-  ``('user', 'follow', 'user')``
-  ``('user', 'followed-by', 'user')``
-  ``('user', 'click', 'item')``
-  ``('item', 'clicked-by', 'user')``
-  ``('user', 'dislike', 'item')``
-  ``('item', 'disliked-by', 'user')``

.. code:: python

    import numpy as np
    import torch
    
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10
    
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)
    
    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})
    
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


Roadmap

本章路线图
------------

The chapter has four sections, each for one type of graph learning tasks.

本章共有四节，每节对应一种类型的图学习任务。

* :ref:`guide_cn-training-node-classification`
* :ref:`guide_cn-training-edge-classification`
* :ref:`guide_cn-training-link-prediction`
* :ref:`guide_cn-training-graph-classification`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    training-node
    training-edge
    training-link
    training-graph
