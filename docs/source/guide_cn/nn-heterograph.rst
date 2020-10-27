.. _guide_cn-nn-heterograph:

3.3 异构图上的GraphConv模块
--------------------------------

:ref:`(English Version) <guide-nn-heterograph>`

DGL的 :class:`~dgl.nn.pytorch.HeteroGraphConv` 是模块级封装，用于在异构图上运行多个不同的DGL NN模块。
实现逻辑与消息传递级别的API :meth:`~dgl.DGLGraph.multi_update_all` 相同，它包括：

-  每个关系 :math:`r` 上的DGL NN模块。
-  合并来自多个关系的相同节点类型上的结果的聚合方式。

这可以表述为：

.. math::  h_{dst}^{(l+1)} = \underset{r\in\mathcal{R}, r_{dst}=dst}{AGG} (f_r(g_r, h_{r_{src}}^l, h_{r_{dst}}^l))

其中 :math:`f_r` 是对应每个关系 :math:`r` 的NN模块，:math:`AGG` 是聚合函数。

HeteroGraphConv的实现逻辑
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code::

    import torch.nn as nn

    class HeteroGraphConv(nn.Module):
        def __init__(self, mods, aggregate='sum'):
            super(HeteroGraphConv, self).__init__()
            self.mods = nn.ModuleDict(mods)
            if isinstance(aggregate, str):
                # 获取聚合函数的内部函数
                self.agg_fn = get_aggregate_fn(aggregate)
            else:
                self.agg_fn = aggregate

异构图的卷积操作接受1个字典 ``mods``，这个字典将每个关系映射到1个NN模块。然后设置将来自多个关系的结果聚合到相同节点类型的函数。

.. code::

    def forward(self, g, inputs, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty : [] for nty in g.dsttypes}

除了输入图和输入张量，``forward()`` 函数还使用2个额外的字典参数 ``mod_args`` 和 ``mod_kwargs``。
这2个字典与 ``self.mods`` 具有相同的键。当针对不同类型的关系在 ``self.mods`` 中调用其对应的NN模块时，它们将被用作自定义参数。

1个输出字典(``output``)被创建出来以保存每个目标节点类型 ``nty`` 的输出张量。请注意，每个 ``nty`` 的值是1个列表，
指示如果多个关系中有 ``nty`` 作为目标节点类型，则单个节点类型可能会获得多个输出。它们将会被保留在列表中以让
``HeteroGraphConv`` 进行进一步聚合。

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

输入 ``g`` 可以是异构图或来自异构图的子图区块。和普通的NN模块一样，``forward()`` 函数需要分别处理不同的输入图类型。

每个关系都被表示为1个 ``canonical_etype``，即 ``(stype, etype, dtype)``。使用 ``canonical_etype`` 作为键，
二分图 ``rel_graph`` 可被提取出来。对于二部图，输入特征将被组织为元组 ``(src_inputs[stype], dst_inputs[dtype])``。
然后调用每个关系的NN模块，并保存输出。为了避免不必要的调用，将跳过没有边或没有其源类型的节点的关系。

.. code::

        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                rsts[nty] = self.agg_fn(alist, nty)

最后，使用 ``self.agg_fn`` 函数聚合来自多个关系的相同目标节点类型上的结果。
读者可以在API文档中找到 :class:`~dgl.nn.pytorch.HeteroGraphConv` 的示例。
