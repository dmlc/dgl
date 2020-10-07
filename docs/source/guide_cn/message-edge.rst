.. _guide_cn-message-passing-edge:

2.4 在消息传递中使用边的权重
-----------------------

:ref:`(English Version) <guide-message-passing-edge>`

一类常见的图神经网络建模的做法是在消息聚合前使用边的权重，
比如在 `图注意力网络(GAT) <https://arxiv.org/pdf/1710.10903.pdf>`__ 和一些 `GCN的变种 <https://arxiv.org/abs/2004.00445>`__ 。
DGL的处理方法是：

-  将权重存为边的特征。
-  在消息函数中用边的特征与源节点的特征相乘。

例如：

.. code::

    import dgl.function as fn

    graph.edata['a'] = affinity
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))

在以上代码中，affinity被用作边的权重。边权重通常是一个标量。