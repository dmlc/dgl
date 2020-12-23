.. _guide_cn-minibatch-custom-gnn-module:

6.5 为小批次训练实现定制化的GNN模块
-------------------------------------------------------------

:ref:`(English Version) <guide-minibatch-custom-gnn-module>`

If you were familiar with how to write a custom GNN module for updating
the entire graph for homogeneous or heterogeneous graphs (see
:ref:`guide-nn`), the code for computing on
blocks is similar, with the exception that the nodes are divided into
input nodes and output nodes.

如果读者熟悉如何定制用于更新整个同构图或异构图的GNN模块(参见
:ref:`guide_cn-nn`)，那么在块上计算的代码也是类似的，区别只在于节点被划分为输入节点和输出节点。

For example, consider the following custom graph convolution module
code. Note that it is not necessarily among the most efficient implementations
- they only serve for an example of how a custom GNN module could look
like.

以下面的自定义图卷积模块代码为例。注意，该代码并不一定是最高效的实现，
此处只是将其作为自定义GNN模块的一个示例。

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            with g.local_scope():
                g.ndata['h'] = h
                g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))

If you have a custom message passing NN module for the full graph, and
you would like to make it work for blocks, you only need to rewrite the
forward function as follows. Note that the corresponding statements from
the full-graph implementation are commented; you can compare the
original statements with the new statements.

如果读者有一个用于整个图的自定义消息传递模块，并且想将其用于块，则只需要按照如下的方法重写forward函数。
注意，以下代码在注释里保留了整图实现的语句，读者可以将用于块的语句和原先用于整图的语句进行比较。

.. code:: python

    class CustomGraphConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        # h is now a pair of feature tensors for input and output nodes, instead of
        # a single feature tensor.
        # h现在是输入和输出节点的特征张量对，而不是一个单独的特征张量

        # def forward(self, g, h):
        def forward(self, block, h):
            # with g.local_scope():
            with block.local_scope():
                # g.ndata['h'] = h
                h_src = h
                h_dst = h[:block.number_of_dst_nodes()]
                block.srcdata['h'] = h_src
                block.dstdata['h'] = h_dst
    
                # g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
                block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
    
                # return self.W(torch.cat([g.ndata['h'], g.ndata['h_neigh']], 1))
                return self.W(torch.cat(
                    [block.dstdata['h'], block.dstdata['h_neigh']], 1))

In general, you need to do the following to make your NN module work for
blocks.

-  Obtain the features for output nodes from the input features by
   slicing the first few rows. The number of rows can be obtained by
   :meth:`block.number_of_dst_nodes <dgl.DGLHeteroGraph.number_of_dst_nodes>`.
-  Replace
   :attr:`g.ndata <dgl.DGLHeteroGraph.ndata>` with either
   :attr:`block.srcdata <dgl.DGLHeteroGraph.srcdata>` for features on input nodes or
   :attr:`block.dstdata <dgl.DGLHeteroGraph.dstdata>` for features on output nodes, if
   the original graph has only one node type.
-  Replace
   :attr:`g.nodes <dgl.DGLHeteroGraph.nodes>` with either
   :attr:`block.srcnodes <dgl.DGLHeteroGraph.srcnodes>` for features on input nodes or
   :attr:`block.dstnodes <dgl.DGLHeteroGraph.dstnodes>` for features on output nodes,
   if the original graph has multiple node types.
-  Replace
   :meth:`g.number_of_nodes <dgl.DGLHeteroGraph.number_of_nodes>` with either
   :meth:`block.number_of_src_nodes <dgl.DGLHeteroGraph.number_of_src_nodes>` or
   :meth:`block.number_of_dst_nodes <dgl.DGLHeteroGraph.number_of_dst_nodes>` for the number of
   input nodes or output nodes respectively.

通常，读者需要对用于整图的GNN模块进行如下调整以将其用于块：

-  切片取输入特征的前几行，得到输出节点的特征。切片行数可以通过
   :meth:`block.number_of_dst_nodes <dgl.DGLHeteroGraph.number_of_dst_nodes>` 获得。
-  如果原图只包含一种节点类型，对输入节点特征，将 :attr:`g.ndata <dgl.DGLHeteroGraph.ndata>` 替换为
   :attr:`block.srcdata <dgl.DGLHeteroGraph.srcdata>`；对于输出节点特征，将
   :attr:`g.ndata <dgl.DGLHeteroGraph.ndata>`  替换为
   :attr:`block.dstdata <dgl.DGLHeteroGraph.dstdata>`。
-  如果原图包含多种节点类型，对于输入节点特征，将
   :attr:`g.nodes <dgl.DGLHeteroGraph.nodes>` 替换为
   :attr:`block.srcnodes <dgl.DGLHeteroGraph.srcnodes>`；对于输出节点特征，将
   :attr:`g.nodes <dgl.DGLHeteroGraph.nodes>` 替换为
   :attr:`block.dstnodes <dgl.DGLHeteroGraph.dstnodes>`。
-  对于输入节点数量，将 :meth:`g.number_of_nodes <dgl.DGLHeteroGraph.number_of_nodes>` 替换为
   :meth:`block.number_of_src_nodes <dgl.DGLHeteroGraph.number_of_src_nodes>` ；
   对于输出节点数量，将 :meth:`g.number_of_nodes <dgl.DGLHeteroGraph.number_of_nodes>` 替换为
   :meth:`block.number_of_dst_nodes <dgl.DGLHeteroGraph.number_of_dst_nodes>` 。

Heterogeneous graphs

异构图上的模型定制
~~~~~~~~~~~~~~~~~~~~

For heterogeneous graph the way of writing custom GNN modules is
similar. For instance, consider the following module that work on full
graph.

为异构图实现定制化的GNN模块的方法是类似的。例如，考虑以下用于全图的GNN模块：

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    g.nodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.nodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.nodes[vtype].data['h_dst'] = g.nodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.nodes[vtype].data['h_neigh'])
                return {ntype: g.nodes[ntype].data['h_dst'] for ntype in g.ntypes}

For ``CustomHeteroGraphConv``, the principle is to replace ``g.nodes``
with ``g.srcnodes`` or ``g.dstnodes`` depend on whether the features
serve for input or output.

对于 ``CustomHeteroGraphConv``，原则是将 ``g.nodes`` 替换为 ``g.srcnodes`` 或
``g.dstnodes`` (根据需要输入还是输出节点的特征来选择)。

.. code:: python

    class CustomHeteroGraphConv(nn.Module):
        def __init__(self, g, in_feats, out_feats):
            super().__init__()
            self.Ws = nn.ModuleDict()
            for etype in g.canonical_etypes:
                utype, _, vtype = etype
                self.Ws[etype] = nn.Linear(in_feats[utype], out_feats[vtype])
            for ntype in g.ntypes:
                self.Vs[ntype] = nn.Linear(in_feats[ntype], out_feats[ntype])
    
        def forward(self, g, h):
            with g.local_scope():
                for ntype in g.ntypes:
                    h_src, h_dst = h[ntype]
                    g.dstnodes[ntype].data['h_dst'] = self.Vs[ntype](h[ntype])
                    g.srcnodes[ntype].data['h_src'] = h[ntype]
                for etype in g.canonical_etypes:
                    utype, _, vtype = etype
                    g.update_all(
                        fn.copy_u('h_src', 'm'), fn.mean('m', 'h_neigh'),
                        etype=etype)
                    g.dstnodes[vtype].data['h_dst'] = \
                        g.dstnodes[vtype].data['h_dst'] + \
                        self.Ws[etype](g.dstnodes[vtype].data['h_neigh'])
                return {ntype: g.dstnodes[ntype].data['h_dst']
                        for ntype in g.ntypes}

Writing modules that work on homogeneous graphs, bipartite graphs, and blocks

实现能够处理同构图、二部图和块的模块
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

All message passing modules in DGL work on homogeneous graphs,
unidirectional bipartite graphs (that have two node types and one edge
type), and a block with one edge type. Essentially, the input graph and
feature of a builtin DGL neural network module must satisfy either of
the following cases.

DGL中所有的消息传递模块都能够处理同构图、单向二部图(包含两种节点类型和一种边类型)和包含一种边类型的块。
本质上，内置的DGL神经网络模块的输入图及特征必须满足下列情况之一：

-  If the input feature is a pair of tensors, then the input graph must
   be unidirectional bipartite.
-  If the input feature is a single tensor and the input graph is a
   block, DGL will automatically set the feature on the output nodes as
   the first few rows of the input node features.
-  If the input feature must be a single tensor and the input graph is
   not a block, then the input graph must be homogeneous.

-  如果输入特征是一个张量对，则输入图必须是一个单向二部图
-  如果输入特征是一个单独的张量且输入图是一个块，则DGL会自动将输入节点特征前一部分设为输出节点的特征。
-  如果输入特征是一个单独的张量且输入图不是块，则输入图必须是同构图。

For example, the following is simplified from the PyTorch implementation
of :class:`dgl.nn.pytorch.SAGEConv` (also available in MXNet and Tensorflow)
(removing normalization and dealing with only mean aggregation etc.).

例如，下面的代码是 :class:`dgl.nn.pytorch.SAGEConv` 的简化版(也适用于MXNet和TensorFlow)。
代码里移除了归一化，且只考虑平均聚合函数的情况。

.. code:: python

    import dgl.function as fn
    class SAGEConv(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.W = nn.Linear(in_feats * 2, out_feats)
    
        def forward(self, g, h):
            if isinstance(h, tuple):
                h_src, h_dst = h
            elif g.is_block:
                h_src = h
                h_dst = h[:g.number_of_dst_nodes()]
            else:
                h_src = h_dst = h
                 
            g.srcdata['h'] = h_src
            g.dstdata['h'] = h_dst
            g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_neigh'))
            return F.relu(
                self.W(torch.cat([g.dstdata['h'], g.dstdata['h_neigh']], 1)))

:ref:`guide-nn` also provides a walkthrough on :class:`dgl.nn.pytorch.SAGEConv`,
which works on unidirectional bipartite graphs, homogeneous graphs, and blocks.

:ref:`guide_cn-nn` 也提供了对 :class:`dgl.nn.pytorch.SAGEConv` 代码的详细解读，
其适用于单向二部图、同构图和块。
