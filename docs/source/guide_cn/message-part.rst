.. _guide_cn-message-passing-part:

2.3 Apply Message Passing On Part Of The Graph

2.3 在图的一部分上进行消息传递
-------------------------

If one only wants to update part of the nodes in the graph, the practice
is to create a subgraph by providing the IDs for the nodes to
include in the update, then call :meth:`~dgl.DGLGraph.update_all` on the
subgraph. For example:

如果用户只想更新图中的部分节点，可以先通过想要更新的节点编号创建一个子图，
然后在子图上调用 :meth:`~dgl.DGLGraph.update_all` 方法。例如：

.. code::

    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func)

This is a common usage in mini-batch training. Check :ref:`guide-minibatch` for more detailed
usages.

这是小批量训练中的常见用法。更多详细用法请参考用户指南 :ref:`第6章：在大图上的随机（批次）训练 <guide-minibatch>`。