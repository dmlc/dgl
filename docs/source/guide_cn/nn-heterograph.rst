.. _guide-nn-heterograph:

3.3 异构图上的GraphConv模块
-------------------------

:class:`~dgl.nn.pytorch.HeteroGraphConv`
is a module-level encapsulation to run DGL NN module on heterogeneous
graphs. The implementation logic is the same as message passing level API
:meth:`~dgl.DGLGraph.multi_update_all`:

dgl.nn.pytorch.HeteroGraphConv是模块级封装，用于在异构图上运行DGL NN模块。实现逻辑与消息传递级别API multi_update_all()相同：

-  DGL NN module within each relation :math:`r`.
-  Reduction that merges the results on the same node type from multiple
   relations.

● 每个关系r中的DGL NN模块。
● 合并来自多个关系的相同节点类型上的结果的聚合方式。

This can be formulated as:

这可以表述为：

.. math::  h_{dst}^{(l+1)} = \underset{r\in\mathcal{R}, r_{dst}=dst}{AGG} (f_r(g_r, h_{r_{src}}^l, h_{r_{dst}}^l))

where :math:`f_r` is the NN module for each relation :math:`r`,
:math:`AGG` is the aggregation function.

其中fr是每个关系r的NN模块，AGG是聚合函数。

HeteroGraphConv implementation logic:

HeteroGraphConv的实现逻辑：
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    import torch.nn as nn

    class HeteroGraphConv(nn.Module):
        def __init__(self, mods, aggregate='sum'):
            super(HeteroGraphConv, self).__init__()
            self.mods = nn.ModuleDict(mods)
            if isinstance(aggregate, str):
                # An internal function to get common aggregation functions
                self.agg_fn = get_aggregate_fn(aggregate)
            else:
                self.agg_fn = aggregate

The heterograph convolution takes a dictionary ``mods`` that maps each
relation to an nn module and sets the function that aggregates results on
the same node type from multiple relations.

异构图的卷积操作采用1个字典mods，将每个关系映射到1个nn模块。并设置将来自多个关系的结果聚合到相同节点类型的函数。

.. code::

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}

Besides input graph and input tensors, the ``forward()`` function takes
two additional dictionary parameters ``mod_args`` and ``mod_kwargs``.
These two dictionaries have the same keys as ``self.mods``. They are
used as customized parameters when calling their corresponding NN
modules in ``self.mods`` for different types of relations.

除了输入图和输入张量，forward()函数还使用2个额外的字典参数mod_args和mod_kwargs。
这2个字典与self.mods具有相同的键。当针对不同类型的关系在self.mods中调用其相应的NN模块时，它们将用作自定义参数。

An output dictionary is created to hold output tensor for each
destination type ``nty`` . Note that the value for each ``nty`` is a
list, indicating a single node type may get multiple outputs if more
than one relations have ``nty`` as the destination type. ``HeteroGraphConv``
will perform a further aggregation on the lists.

创建1个输出字典来保存每个目标类型nty的输出张量。请注意，每个nty的值是1个列表，
指示如果多个关系中有nty作为目标类型，则单个节点类型可能会获得多个输出。它们将会被保留在列表中以进行进一步聚合。

.. code::

          if g.is_block:
              src_inputs = inputs
              dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
          else:
              src_inputs = dst_inputs = inputs

          for stype, etype, dtype in g.canonical_etypes:
              rel_graph = g[stype, etype, dtype]
              if rel_graph.num_edges() == 0:
                  continue
              if stype not in src_inputs or dtype not in dst_inputs:
                  continue
              dstdata = self.mods[etype](
                  rel_graph,
                  (src_inputs[stype], dst_inputs[dtype]),
                  *mod_args.get(etype, ()),
                  **mod_kwargs.get(etype, {}))
              outputs[dtype].append(dstdata)

The input ``g`` can be a heterogeneous graph or a subgraph block from a
heterogeneous graph. As in ordinary NN module, the ``forward()``
function need to handle different input graph types separately.

输入g可以是异构图或来自异构图的子图块。和普通的NN模块一样，forward()函数需要分别处理不同的输入图类型。

Each relation is represented as a ``canonical_etype``, which is
``(stype, etype, dtype)``. Using ``canonical_etype`` as the key, one can
extract out a bipartite graph ``rel_graph``. For bipartite graph, the
input feature will be organized as a tuple
``(src_inputs[stype], dst_inputs[dtype])``. The NN module for each
relation is called and the output is saved. To avoid unnecessary call,
relations with no edges or no nodes with the src type will be skipped.

每个关系都表示为1个canonical_etype，即(stype, etype, dtype)。使用canonical_etype作为键，
二部图rel_graph可被提取出来。对于二部图，输入特征将被组织为元组(src_inputs[stype], dst_inputs[dtype])。
调用每个关系的NN模块，并保存输出。为了避免不必要的调用，将跳过没有边或没有其src类型的节点的关系。

.. code::

        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)

Finally, the results on the same destination node type from multiple
relations are aggregated using ``self.agg_fn`` function. Examples can
be found in the API Doc for :class:`~dgl.nn.pytorch.HeteroGraphConv`.

最后，使用self.agg_fn函数聚合来自多个关系的相同目标节点类型上的结果。
可以在API文档中找到dgl.nn.pytorch.HeteroGraphConv的示例。
