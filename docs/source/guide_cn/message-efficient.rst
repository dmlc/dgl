.. _guide_cn-message-passing-efficient:

2.2 编写高效的消息传递代码
----------------------

DGL optimizes memory consumption and computing speed for message
passing. The optimization includes:

DGL优化了消息传递的内存消耗和计算速度，这包括：


-  Merge multiple kernels in a single one: This is achieved by using
   :meth:`~dgl.DGLGraph.update_all` to call multiple built-in functions
   at once. (Speed optimization)

-  将多个内核合并到一个内核中：这是通过使用 :meth:`~dgl.DGLGraph.update_all` 一次调用多个内置函数来实现的。（速度优化）

-  Parallelism on nodes and edges: DGL abstracts edge-wise computation
   :meth:`~dgl.DGLGraph.apply_edges` as a generalized sampled dense-dense
   matrix multiplication (**gSDDMM**) operation and parallelizes the computing
   across edges. Likewise, DGL abstracts node-wise computation
   :meth:`~dgl.DGLGraph.update_all` as a generalized sparse-dense matrix
   multiplication (**gSPMM**) operation and parallelizes the computing across
   nodes. (Speed optimization)

-  节点和边上的并行性：DGL抽象了逐边计算，将 :meth:`~dgl.DGLGraph.apply_edges` 作为一种广义抽样稠密-稠密矩阵乘法
   **（gSDDMM）** 运算，并实现了跨边并行计算。同样，DGL将逐节点计算 :meth:`~dgl.DGLGraph.update_all` 抽象为广义稀疏-稠密矩阵乘法（gSPMM）运算，
   并实现了跨节点并行计算。（速度优化）

-  Avoid unnecessary memory copy from nodes to edges: To generate a
   message that requires the feature from source and destination node,
   one option is to copy the source and destination node feature to
   that edge. For some graphs, the number of edges is much larger than
   the number of nodes. This copy can be costly. DGL's built-in message
   functions avoid this memory copy by sampling out the node feature using
   entry index. (Memory and speed optimization)

-  避免不必要的从点到边的内存拷贝：要生成需要源节点和目标节点特征的消息，一个选项是将源节点和目标节点特征拷贝到边上。
   对于某些图，边的数目远远大于节点的数目。这个拷贝的代价会很大。DGL内置的消息函数通过使用条目索引对节点特征进行采集来避免这种内存拷贝。
   （内存和速度优化）

-  Avoid materializing feature vectors on edges: the complete message
   passing process includes message generation, message aggregation and
   node update. In :meth:`~dgl.DGLGraph.update_all` call, message function
   and reduce function are merged into one kernel if those functions are
   built-in. There is no message materialization on edges. (Memory
   optimization)

-  避免具体化边上的特征向量：完整的消息传递过程包括消息生成、消息聚合和节点更新。
   在调用 :meth:`~dgl.DGLGraph.update_all` 时，如果消息函数和聚合函数是内置的，则它们会被合并到一个内核中。
   边上的消息没有被具体化。（内存优化）

According to the above, a common practise to leverage those
optimizations is to construct one's own message passing functionality as
a combination of :meth:`~dgl.DGLGraph.update_all` calls with built-in
functions as parameters.

根据以上所述，利用这些优化的一个常见实践是通过基于内置的 :meth:`~dgl.DGLGraph.update_all` 来开发消息传递功能。

For some cases like
:class:`~dgl.nn.pytorch.conv.GATConv`,
where it is necessary to save message on the edges, one needs to call
:meth:`~dgl.DGLGraph.apply_edges` with built-in functions. Sometimes the
messages on the edges can be high dimensional, which is memory consuming.
DGL recommends keeping the dimension of edge features as low as possible.

对于某些情况，比如 :class:`~dgl.nn.pytorch.conv.GATConv`，计算必须在边上保存消息，
用户需要调用基于内置函数的:meth:`~dgl.DGLGraph.apply_edges` 。有时边上的消息可能是高维的，这会非常消耗内存。
DGL建议用户尽可能降低edata维数。

Here’s an example on how to achieve this by splitting operations on the
edges to nodes. The approach does the following: concatenate the ``src``
feature and ``dst`` feature, then apply a linear layer, i.e.
:math:`W\times (u || v)`. The ``src`` and ``dst`` feature dimension is
high, while the linear layer output dimension is low. A straight forward
implementation would be like:

下面是一个如何通过对节点特征降维来减少消息维度的示例。该做法执行以下操作：拼接 ``源`` 节点和 ``目标`` 节点特征，
然后应用一个线性层，即 :math:`W\times (u || v)`。 ``源`` 节点和 ``目标`` 节点特征维数较高，而线性层输出维数较低。
一个直截了当的实现方式如下：

.. code::

    import torch
    import torch.nn as nn

    linear = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim * 2)))
    def concat_message_function(edges):
         return {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']])}
    g.apply_edges(concat_message_function)
    g.edata['out'] = g.edata['cat_feat'] * linear

The suggested implementation splits the linear operation into two,
one applies on ``src`` feature, the other applies on ``dst`` feature.
It then adds the output of the linear operations on the edges at the final stage,
i.e. performing :math:`W_l\times u + W_r \times v`. This is because
:math:`W \times (u||v) = W_l \times u + W_r \times v`, where :math:`W_l`
and :math:`W_r` are the left and the right half of the matrix :math:`W`,
respectively:

建议的实现将线性操作分成两部分，一个应用于 ``源`` 节点特征，另一个应用于 ``目标`` 节点特征。
在最后一个阶段，在边上将以上两部分线性操作的结果相加，即执行 :math:`W_l\times u + W_r \times v`，
因为 :math:`W \times (u||v) = W_l \times u + W_r \times v`，其中 :math:`W_l` 和 :math:`W_r`分别是矩阵 :math:`W` 的左半部分和右半部分：

.. code::

    import dgl.function as fn

    linear_src = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim)))
    linear_dst = nn.Parameter(torch.FloatTensor(size=(1, node_feat_dim)))
    out_src = g.ndata['feat'] * linear_src
    out_dst = g.ndata['feat'] * linear_dst
    g.srcdata.update({'out_src': out_src})
    g.dstdata.update({'out_dst': out_dst})
    g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))

The above two implementations are mathematically equivalent. The latter
one is more efficient because it does not need to save feat_src and
feat_dst on edges, which is not memory-efficient. Plus, addition could
be optimized with DGL’s built-in function ``u_add_v``, which further
speeds up computation and saves memory footprint.

以上两个实现在数学上是等价的。后一种方法效率高得多，因为不需要在边上保存feat_src和feat_dst，
而这从内存角度来说是低效的。另外，加法可以通过DGL的内置函数 ``u_add_v`` 进行优化，从而进一步加快计算速度并节省内存占用。
