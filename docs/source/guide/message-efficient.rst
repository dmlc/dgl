.. _guide-message-passing-efficient:

2.2 Writing Efficient Message Passing Code
------------------------------------------

:ref:`(中文版) <guide_cn-message-passing-efficient>`

DGL optimizes memory consumption and computing speed for message
passing. The optimization includes:

-  Merge multiple kernels in a single one: This is achieved by using
   :meth:`~dgl.DGLGraph.update_all` to call multiple built-in functions
   at once. (Speed optimization)

-  Parallelism on nodes and edges: DGL abstracts edge-wise computation
   :meth:`~dgl.DGLGraph.apply_edges` as a generalized sampled dense-dense
   matrix multiplication (**gSDDMM**) operation and parallelizes the computing
   across edges. Likewise, DGL abstracts node-wise computation
   :meth:`~dgl.DGLGraph.update_all` as a generalized sparse-dense matrix
   multiplication (**gSPMM**) operation and parallelizes the computing across
   nodes. (Speed optimization)

-  Avoid unnecessary memory copy from nodes to edges: To generate a
   message that requires the feature from source and destination node,
   one option is to copy the source and destination node feature to
   that edge. For some graphs, the number of edges is much larger than
   the number of nodes. This copy can be costly. DGL's built-in message
   functions avoid this memory copy by sampling out the node feature using
   entry index. (Memory and speed optimization)

-  Avoid materializing feature vectors on edges: the complete message
   passing process includes message generation, message aggregation and
   node update. In :meth:`~dgl.DGLGraph.update_all` call, message function
   and reduce function are merged into one kernel if those functions are
   built-in. There is no message materialization on edges. (Memory
   optimization)

According to the above, a common practise to leverage those
optimizations is to construct one's own message passing functionality as
a combination of :meth:`~dgl.DGLGraph.update_all` calls with built-in
functions as parameters.

For some cases like
:class:`~dgl.nn.pytorch.conv.GATConv`,
where it is necessary to save message on the edges, one needs to call
:meth:`~dgl.DGLGraph.apply_edges` with built-in functions. Sometimes the
messages on the edges can be high dimensional, which is memory consuming.
DGL recommends keeping the dimension of edge features as low as possible.

Here’s an example on how to achieve this by splitting operations on the
edges to nodes. The approach does the following: concatenate the ``src``
feature and ``dst`` feature, then apply a linear layer, i.e.
:math:`W\times (u || v)`. The ``src`` and ``dst`` feature dimension is
high, while the linear layer output dimension is low. A straight forward
implementation would be like:

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
i.e. performing :math:`W_l\times u + W_r \times v`. This is because
:math:`W \times (u||v) = W_l \times u + W_r \times v`, where :math:`W_l`
and :math:`W_r` are the left and the right half of the matrix :math:`W`,
respectively:

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
