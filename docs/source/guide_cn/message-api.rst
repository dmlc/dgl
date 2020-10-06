.. _guide_cn-message-passing-api:

2.1 内置函数和消息传递API
----------------------

:ref:`(English Version) <guide-message-passing-api>`

In DGL, **message function** takes a single argument ``edges``,
which is an :class:`~dgl.udf.EdgeBatch` instance. During message passing,
DGL generates it internally to represent a batch of edges. It has three
members ``src``, ``dst`` and ``data`` to access features of source nodes,
destination nodes, and edges, respectively.

在DGL中，**消息函数** 接受一个参数 ``edges``，它是一个 :class:`~dgl.udf.EdgeBatch` 的实例，
在消息传递时，DGL在内部生成以表示一批边。这些边有 ``src``、 ``dst`` 和 ``data`` 三个成员属性，
分别可以用于访问源节点、目标节点和边的特征。

**reduce function** takes a single argument ``nodes``, which is a
:class:`~dgl.udf.NodeBatch` instance. During message passing,
DGL generates it internally to represent a batch of nodes. It has member
``mailbox`` to access the messages received for the nodes in the batch.
Some of the most common reduce operations include ``sum``, ``max``, ``min``, etc.

**聚合函数** 接受一个参数 ``nodes``，它是一个 :class:`~dgl.udf.NodeBatch` 的实例，
在消息传递时，DGL在内部生成以表示一批节点。这些节点的成员属性 ``mailbox`` 可以用来访问节点收到的消息。
一些最常见的聚合操作包括 ``sum``、``max``、``min`` 等。

**update function** takes a single argument ``nodes`` as described above.
This function operates on the aggregation result from ``reduce function``, typically
combining it with a node’s original feature at the the last step and saving the result
as a node feature.

**更新函数** 接受一个参数 ``nodes``。此函数对 ``聚合函数`` 的聚合结果进行操作，
通常在消息传递的最后一步将其与节点的特征相结合，并将输出为节点特征。

DGL has implemented commonly used message functions and reduce functions
as **built-in** in the namespace ``dgl.function``. In general, DGL
suggests using built-in functions **whenever possible** since they are
heavily optimized and automatically handle dimension broadcasting.

DGL在命名空间 ``dgl.function`` 中实现了常用的消息函数和聚合函数作为 **内置函数**。
一般来说，DGL建议 **尽可能** 使用内置函数，因为它们经过了大量优化，并且可以自动处理维度广播。

If your message passing functions cannot be implemented with built-ins,
you can implement user-defined message/reduce function (aka. **UDF**).

如果用户的消息传递函数不能用内置函数实现，用户可以实现自己的消息或聚合函数(也称为 **用户定义函数** )。

Built-in message functions can be unary or binary. DGL supports ``copy``
for unary. For binary funcs, DGL supports ``add``, ``sub``, ``mul``, ``div``,
``dot``. The naming convention for message built-in funcs is that ``u``
represents ``src`` nodes, ``v`` represents ``dst`` nodes, and ``e`` represents ``edges``.
The parameters for those functions are strings indicating the input and output field names for
the corresponding nodes and edges. The list of supported built-in functions
can be found in :ref:`api-built-in`. For example, to add the ``hu`` feature from src
nodes and ``hv`` feature from dst nodes then save the result on the edge
at ``he`` field, one can use built-in function ``dgl.function.u_add_v('hu', 'hv', 'he')``.
This is equivalent to the Message UDF:

内置消息函数可以是一元函数或二元函数。对于一元函数，DGL支持 ``copy`` 函数。对于二元函数，
DGL现在支持 ``add``、 ``sub``、 ``mul``、 ``div``、 ``dot``。消息内置函数的命名约定是 ``u`` 表示 ``源`` 节点，
``v`` 表示 ``目标`` 节点，``e``表示 ``边``。这些函数的参数是字符串，指示相应节点和边的输入和输出特征字段名。
关于内置函数，见 :ref:`api-built-in`。例如，要对源节点的 ``hu`` 特征和目标节点的 ``hv`` 特征求和，
然后将结果保存在边的 ``he`` 特征上，用户可以使用内置函数 ``dgl.function.u_add_v('hu', 'hv', 'he')``。
而以下用户定义消息函数与此内置函数等价。

.. code::

    def message_func(edges):
         return {'he': edges.src['hu'] + edges.dst['hv']}

Built-in reduce functions support operations ``sum``, ``max``, ``min``,
and ``mean``. Reduce functions usually have two parameters, one
for field name in ``mailbox``, one for field name in node features, both
are strings. For example, ``dgl.function.sum('m', 'h')`` is equivalent
to the Reduce UDF that sums up the message ``m``:

内置的聚合函数支持 ``sum``、 ``max``、 ``min``、 ``prod`` 和 ``mean`` 操作。
聚合函数通常有两个参数，它们的类型都是字符串。一个用于指定 ``mailbox`` 中的字段名，一个用于指示目标节点特征的字段名，
例如， ``dgl.function.sum('m', 'h')`` 等价于如下所示的对接收到消息求和的用户定义函数：

.. code::

    import torch
    def reduce_func(nodes):
         return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

It is also possible to invoke only edge-wise computation by :meth:`~dgl.DGLGraph.apply_edges`
without invoking message passing. :meth:`~dgl.DGLGraph.apply_edges` takes a message function
for parameter and by default updates the features of all edges. For example:

在DGL中，也可以调用逐边计算的接口 :meth:`~dgl.DGLGraph.apply_edges`，而不必显式地调用消息传递函数。
:meth:`~dgl.DGLGraph.apply_edges` 的参数是一个消息函数，并且在默认情况下，这个接口将更新所有的边。例如：

.. code::

    import dgl.function as fn
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

For message passing, :meth:`~dgl.DGLGraph.update_all` is a high-level
API that merges message generation, message aggregation and node update
in a single call, which leaves room for optimization as a whole.

对于消息传递， :meth:`~dgl.DGLGraph.update_all` 是一个高级API。它聚合了消息生成、
消息聚合和节点特征更新为一体，从而能从整体上进行系统优化。

The parameters for :meth:`~dgl.DGLGraph.update_all` are a message function, a
reduce function and an update function. One can call update function outside of
``update_all`` and not specify it in invoking :meth:`~dgl.DGLGraph.update_all`.
DGL recommends this approach since the update function can usually be
written as pure tensor operations to make the code concise. For
example：

:meth:`~dgl.DGLGraph.update_all` 的参数是一个消息函数、一个聚合函数和一个更新函数。
更新函数是一个选择性的参数。用户也可在 ``update_all`` 执行完后直接对节点特征进行操做。
由于更新函数通常可以以纯张量操作实现，DGL不推荐在 ``update_all`` 中指定更新函数，
而是在它执行完后直接对节点特征进行操作。例如：

.. code::

    def updata_all_example(graph):
        # store the result in graph.ndata['ft']
        # 在 graph.ndata['ft']中存储结果
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        # Call update function outside of update_all
        # 在update_all外调用更新函数
        final_ft = graph.ndata['ft'] * 2
        return final_ft

This call will generate the messages ``m`` by multiply src node features
``ft`` and edge features ``a``, sum up the messages ``m`` to update node
features ``ft``, and finally multiply ``ft`` by 2 to get the result
``final_ft``. After the call, DGL will clean the intermediate messages ``m``.
The math formula for the above function is:

此调用通过将源节点特征 ``ft`` 与边特征 ``a`` 相乘生成消息 ``m``，
然后对所有消息求和来更新节点特征 ``ft``，最后将 ``ft`` 乘以2得到最终结果 ``final_ft``。

调用后，中间消息 ``m`` 将被清除。上述函数的数学公式为：

.. math::  {final\_ft}_i = 2 * \sum_{j\in\mathcal{N}(i)} ({ft}_j * a_{ij})
