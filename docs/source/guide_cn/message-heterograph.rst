.. _guide_cn-message-passing-heterograph:

2.5 在异构图上进行消息传递
----------------------

:ref:`(English Version) <guide-message-passing-heterograph>`

Heterogeneous graphs (:ref:`guide-graph-heterogeneous`), or
heterographs for short, are graphs that contain different types of nodes
and edges. The different types of nodes and edges tend to have different
types of attributes that are designed to capture the characteristics of
each node and edge type. Within the context of graph neural networks,
depending on their complexity, certain node and edge types might need to
be modeled with representations that have a different number of
dimensions.

异构图（参考用户指南 :ref:`第一章1.5 异构图 <guide_cn-graph-heterogeneous>` ）是包含不同类型的节点和边的图。
不同类型的节点和边常常具有不同类型的属性。这些属性旨在刻画每一种节点和边的特征。在使用图神经网络时，根据其复杂性，
可能需要使用不同维数的表示来对不同类型的节点和边进行建模。

The message passing on heterographs can be split into two parts:

异构图上的消息传递可以分为2个部分：

1. Message computation and aggregation for each relation r.
2. Reduction that merges the aggregation results from all relations for each node type.

1. 对每个关系r计算和聚合消息。
2. 对每个节点类型整合不同关系下聚合的消息。

DGL’s interface to call message passing on heterographs is
:meth:`~dgl.DGLGraph.multi_update_all`.
:meth:`~dgl.DGLGraph.multi_update_all` takes a dictionary containing
the parameters for :meth:`~dgl.DGLGraph.update_all` within each relation
using relation as the key, and a string representing the cross type reducer.
The reducer can be one of ``sum``, ``min``, ``max``, ``mean``, ``stack``.
Here’s an example:

在DGL中，对异构图进行消息传递的接口是 :meth:`~dgl.DGLGraph.multi_update_all`。
:meth:`~dgl.DGLGraph.multi_update_all` 接受一个字典。这个字典的每一个键值对里，键是一种关系，
值是这种关系对应 :meth:`~dgl.DGLGraph.update_all` 的参数。
:meth:`~dgl.DGLGraph.multi_update_all` 还接受一个字符串来表示跨类型整合函数，
这个整合函数可以是 ``sum``， ``min``， ``max``， ``mean`` 和 ``stack`` 中的一个。以下是一个例子：

.. code::

    import dgl.function as fn

    for c_etype in G.canonical_etypes:
        srctype, etype, dsttype = c_etype
        Wh = self.weight[etype](feat_dict[srctype])
        # Save it in graph for message passing
        # 把它存在图中用来做消息传递
        G.nodes[srctype].data['Wh_%s' % etype] = Wh
        # Specify per-relation message passing functions: (message_func, reduce_func).
        # Note that the results are saved to the same destination feature 'h', which
        # hints the type wise reducer for aggregation.
        # 指定每个关系的消息传递函数：(message_func, reduce_func).
        # 注意结果保存在同一个目标特征“h”，说明聚合是逐类进行的。
        funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
    # Trigger message passing of multiple types.
    # 将每个类型消息聚合的结果相加。
    G.multi_update_all(funcs, 'sum')
    # return the updated node feature dictionary
    # 返回更新过的节点特征字典
    return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
