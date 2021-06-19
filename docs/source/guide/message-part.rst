.. _guide-message-passing-part:

2.3 Apply Message Passing On Part Of The Graph
----------------------------------------------

:ref:`(中文版) <guide_cn-message-passing-part>`

If one only wants to update part of the nodes in the graph, the practice
is to create a subgraph by providing the IDs for the nodes to
include in the update, then call :meth:`~dgl.DGLGraph.update_all` on the
subgraph. For example:

.. code::

    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func)

This is a common usage in mini-batch training. Check :ref:`guide-minibatch` for more detailed
usages.