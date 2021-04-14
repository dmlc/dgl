.. _guide_cn-message-passing-part:

2.3 在图的一部分上进行消息传递
-------------------------

:ref:`(English Version) <guide-message-passing-part>`

如果用户只想更新图中的部分节点，可以先通过想要囊括的节点编号创建一个子图，
然后在子图上调用 :meth:`~dgl.DGLGraph.update_all` 方法。例如：

.. code::

    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func)

这是小批量训练中的常见用法。更多详细用法请参考用户指南 :ref:`guide_cn-minibatch`。