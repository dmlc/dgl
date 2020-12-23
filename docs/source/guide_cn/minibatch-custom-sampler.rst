.. _guide_cn-minibatch-customizing-neighborhood-sampler:

6.4 定制用户自己的邻居采样器
----------------------------------------------

:ref:`(English Version) <guide-minibatch-customizing-neighborhood-sampler>`

Although DGL provides some neighborhood sampling strategies, sometimes
users would want to write their own sampling strategy. This section
explains how to write your own strategy and plug it into your stochastic
GNN training framework.

虽然DGL提供了一些邻居采样器，但有时用户还是希望编写自己的采样器。
本节说明如何编写自己的采样器并将其加入到GNN随机训练框架中。

Recall that in `How Powerful are Graph Neural
Networks <https://arxiv.org/pdf/1810.00826.pdf>`__, the definition of message
passing is:

回想一下在
`How Powerful are Graph Neural Networks <https://arxiv.org/pdf/1810.00826.pdf>`__
的论文中，消息传递的定义是：

.. math::

   \begin{gathered}
     \boldsymbol{a}_v^{(l)} = \rho^{(l)} \left(
       \left\lbrace
         \boldsymbol{h}_u^{(l-1)} : u \in \mathcal{N} \left( v \right)
       \right\rbrace
     \right)
   \\
     \boldsymbol{h}_v^{(l)} = \phi^{(l)} \left(
       \boldsymbol{h}_v^{(l-1)}, \boldsymbol{a}_v^{(l)}
     \right)
   \end{gathered}

where :math:`\rho^{(l)}` and :math:`\phi^{(l)}` are parameterized
functions, and :math:`\mathcal{N}(v)` is defined as the set of
predecessors (or *neighbors* if the graph is undirected) of :math:`v` on graph
:math:`\mathcal{G}`.

其中， :math:`\rho^{(l)}` 和 :math:`\phi^{(l)}` 是参数化函数，
并且 :math:`\mathcal{N}(v)` 定义为有向图 :math:`\mathcal{G}` 上节点 :math:`v` 的前驱节点(或无向图中的邻居)。

For instance, to perform a message passing for updating the red node in
the following graph:

例如，要执行消息传递以更新下图中的红色节点：

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_0.png
   :alt: Imgur


One needs to aggregate the node features of its neighbors, shown as
green nodes:

需要聚集其邻居的节点特征，如绿色节点所示：

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_1.png
   :alt: Imgur


Neighborhood sampling with pencil and paper

理解邻居采样
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We then consider how multi-layer message passing works for computing the
output of a single node. In the following text we refer to the nodes
whose GNN outputs are to be computed as *seed nodes*.

接下来考虑多层消息传递时如何计算单个节点的输出。在下文中，DGL将需要计算其GNN输出的节点称为 *种子节点* 。

.. code:: python

    import torch
    import dgl
    
    src = torch.LongTensor(
        [0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10,
         1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11])
    dst = torch.LongTensor(
        [1, 2, 3, 3, 3, 4, 5, 5, 6, 5, 8, 6, 8, 9, 8, 11, 11, 10, 11,
         0, 0, 0, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 9, 10])
    g = dgl.graph((src, dst))
    g.ndata['x'] = torch.randn(12, 5)
    g.ndata['y'] = torch.randn(12, 1)

Finding the message passing dependency

找出消息传递的依赖
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider computing with a 2-layer GNN the output of the seed node 8,
colored red, in the following graph:

在下图中，考虑使用2层GNN计算种子节点8（红色点）的输出：

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_2.png
   :alt: Imgur


By the formulation:

公式如下：

.. math::

   \begin{gathered}
     \boldsymbol{a}_8^{(2)} = \rho^{(2)} \left(
       \left\lbrace
         \boldsymbol{h}_u^{(1)} : u \in \mathcal{N} \left( 8 \right)
       \right\rbrace
     \right) = \rho^{(2)} \left(
       \left\lbrace
         \boldsymbol{h}_4^{(1)}, \boldsymbol{h}_5^{(1)},
         \boldsymbol{h}_7^{(1)}, \boldsymbol{h}_{11}^{(1)}
       \right\rbrace
     \right)
   \\
     \boldsymbol{h}_8^{(2)} = \phi^{(2)} \left(
       \boldsymbol{h}_8^{(1)}, \boldsymbol{a}_8^{(2)}
     \right)
   \end{gathered}

We can tell from the formulation that to compute
:math:`\boldsymbol{h}_8^{(2)}` we need messages from node 4, 5, 7 and 11
(colored green) along the edges visualized below.

从公式可以看出，要计算 :math:`\boldsymbol{h}_8^{(2)}`，需要下图中的来自节点4、5、7和11(绿色点)的消息。

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_3.png
   :alt: Imgur


This graph contains all the nodes in the original graph but only the
edges necessary for message passing to the given output nodes. We call
that the *frontier* of the second GNN layer for the red node 8.

该图虽包含初始图中的所有节点，但仅包含消息传递到给定输出节点所需的边。
DGL称它们为红色节点8在第二个GNN层的 *边界* 。

Several functions can be used for generating frontiers. For instance,
:func:`dgl.in_subgraph()` is a function that induces a
subgraph by including all the nodes in the original graph, but only all
the incoming edges of the given nodes. You can use that as a frontier
for message passing along all the incoming edges.

有几个函数可用于生成边界。例如，
:func:`dgl.in_subgraph()` 是一个生成子图的函数，该子图包括初始图中的所有节点和指定节点的入边。
用户可以将其用作沿所有入边传递消息的边界。

.. code:: python

    frontier = dgl.in_subgraph(g, [8])
    print(frontier.all_edges())

For a concrete list, please refer to :ref:`api-subgraph-extraction` and
:ref:`api-sampling`.

相关具体函数，用户可以参考 :ref:`api-subgraph-extraction` 和 :ref:`api-sampling`。

Technically, any graph that has the same set of nodes as the original
graph can serve as a frontier. This serves as the basis for
:ref:`guide-minibatch-customizing-neighborhood-sampler-impl`.

从技术上讲，任何具有与初始图相同的节点的图都可以用作边界。这是
:ref:`guide_cn-minibatch-customizing-neighborhood-sampler-impl`
基础。

The Bipartite Structure for Multi-layer Minibatch Message Passing

多层小批量消息传递的二部图结构
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

However, to compute :math:`\boldsymbol{h}_8^{(2)}` from
:math:`\boldsymbol{h}_\cdot^{(1)}`, we cannot simply perform message
passing on the frontier directly, because it still contains all the
nodes from the original graph. Namely, we only need nodes 4, 5, 7, 8,
and 11 (green and red nodes) as input, as well as node 8 (red node) as output.
Since the number of nodes
for input and output is different, we need to perform message passing on
a small, bipartite-structured graph instead. We call such a
bipartite-structured graph that only contains the necessary input nodes
and output nodes a *block*. The following figure shows the block of the
second GNN layer for node 8.

但是，要从 :math:`\boldsymbol{h}_\cdot^{(1)}` 计算
 :math:`\boldsymbol{h}_8^{(2)}`，DGL不能简单地直接在边界上执行消息传递，
因为它仍然包含初始图中的所有节点。换言之，消息传递只需要节点4、5、7、8和11（绿色和红色节点）作为输入，
以及节点8（红色节点）作为输出。由于用于输入和输出的节点数不同，
因此DGL需要在一个小的二部图上执行消息传递。DGL称这种仅包含必要的输入节点和输出节点的二部结构图为一个 *块* (block)。
下图显示了以节点8为种子节点时第二个GNN层所需的块。

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_4.png
   :alt: Imgur


Note that the output nodes also appear in the input nodes. The reason is
that representations of output nodes from the previous layer are needed
for feature combination after message passing (i.e. :math:`\phi^{(2)}`).

请注意，输出节点也出现在输入节点中。原因是消息传递后的特征组合需要前一层的输出节点表示
(即 :math:`\phi^{(2)}`)。

DGL provides :func:`dgl.to_block` to convert any frontier
to a block where the first argument specifies the frontier and the
second argument specifies the output nodes. For instance, the frontier
above can be converted to a block with output node 8 with the code as
follows.

DGL提供 :func:`dgl.to_block` 以将任何边界转换为块。其中第一个参数指定边界，
第二个参数指定输出节点。例如，可以使用以下代码将上述边界转换为输出节点为8的块。

.. code:: python

    output_nodes = torch.LongTensor([8])
    block = dgl.to_block(frontier, output_nodes)

To find the number of input nodes and output nodes of a given node type,
one can use :meth:`dgl.DGLHeteroGraph.number_of_src_nodes` and
:meth:`dgl.DGLHeteroGraph.number_of_dst_nodes` methods.

要查找给定节点类型的输入节点和输出节点的数量，可以使用
:meth:`dgl.DGLHeteroGraph.number_of_src_nodes`  和
:meth:`dgl.DGLHeteroGraph.number_of_dst_nodes` 方法。

.. code:: python

    num_input_nodes, num_output_nodes = block.number_of_src_nodes(), block.number_of_dst_nodes()
    print(num_input_nodes, num_output_nodes)

The block’s input node features can be accessed via member
:attr:`dgl.DGLHeteroGraph.srcdata` and :attr:`dgl.DGLHeteroGraph.srcnodes`, and
its output node features can be accessed via member
:attr:`dgl.DGLHeteroGraph.dstdata` and :attr:`dgl.DGLHeteroGraph.dstnodes`. The
syntax of ``srcdata``/``dstdata`` and ``srcnodes``/``dstnodes`` are
identical to :attr:`dgl.DGLHeteroGraph.ndata` and
:attr:`dgl.DGLHeteroGraph.nodes` in normal graphs.

可以通过 :attr:`dgl.DGLHeteroGraph.srcdata` 和
:attr:`dgl.DGLHeteroGraph.srcnodes` 访问该块的输入节点特征，
并且可以通过 :attr:`dgl.DGLHeteroGraph.dstdata` 和
:attr:`dgl.DGLHeteroGraph.dstnodes` 访问其输出节点特征。
 ``srcdata``/``dstdata`` 和 ``srcnodes``/``dstnodes``
的语法与常规图中的 :attr:`dgl.DGLHeteroGraph.ndata` 和 :attr:`dgl.DGLHeteroGraph.nodes` 相同。

.. code:: python

    block.srcdata['h'] = torch.randn(num_input_nodes, 5)
    block.dstdata['h'] = torch.randn(num_output_nodes, 5)

If a block is converted from a frontier, which is in turn converted from
a graph, one can directly read the feature of the block’s input and
output nodes via

如果是从图中得到的边界，再由边界转换成块，则可以通过以下方式直接读取块的输入和输出节点的特征。

.. code:: python

    print(block.srcdata['x'])
    print(block.dstdata['y'])

.. raw:: html

   <div class="alert alert-info">

::

   <b>ID Mappings</b>

The original node IDs of the input nodes and output nodes in the block
can be found as the feature ``dgl.NID``, and the mapping from the
block’s edge IDs to the input frontier’s edge IDs can be found as the
feature ``dgl.EID``.

用户可以通过 ``dgl.NID`` 得到块中输入节点和输出节点的初始节点ID，可以通过 ``dgl.EID``
得到边ID到输入边界的边ID的映射。

.. raw:: html

   </div>

**Output Nodes**

**输出节点**

DGL ensures that the output nodes of a block will always appear in the
input nodes. The output nodes will always index firstly in the input
nodes.

DGL确保块的输出节点将始终出现在输入节点中。在输入节点中，输出节点的ID在其它节点之前。

.. code:: python

    input_nodes = block.srcdata[dgl.NID]
    output_nodes = block.dstdata[dgl.NID]
    assert torch.equal(input_nodes[:len(output_nodes)], output_nodes)

As a result, the output nodes must cover all nodes that are the
destination of an edge in the frontier.

因此，输出节点必须包含边界中所有边的目标节点。

For example, consider the following frontier

例如，考虑以下边界

.. figure:: https://data.dgl.ai/asset/image/guide_6_4_5.png
   :alt: Imgur


where the red and green nodes (i.e. node 4, 5, 7, 8, and 11) are all
nodes that is a destination of an edge. Then the following code will
raise an error because the output nodes did not cover all those nodes.

其中红色和绿色节点（即节点4、5、7、8和11）都是某条边的目标节点。
以下代码由于输出节点未覆盖所有这些节点，将会报错。

.. code:: python

    dgl.to_block(frontier2, torch.LongTensor([4, 5]))   # ERROR

However, the output nodes can have more nodes than above. In this case,
we will have isolated nodes that do not have any edge connecting to it.
The isolated nodes will be included in both input nodes and output
nodes.

但是，输出节点可以比以上节点包含更多节点。下例的输出节点包含了没有入边的孤立节点。
输入节点和输出节点将同时包含这些孤立节点。

.. code:: python

    # Node 3 is an isolated node that do not have any edge pointing to it.
    # 节点3是一个孤立节点，没有任何指向它的边.
    block3 = dgl.to_block(frontier2, torch.LongTensor([4, 5, 7, 8, 11, 3]))
    print(block3.srcdata[dgl.NID])
    print(block3.dstdata[dgl.NID])

Heterogeneous Graphs

异构图上的采样
^^^^^^^^^^^^^^^^^^^^

Blocks also work on heterogeneous graphs. Let’s say that we have the
following frontier:

块也可用于异构图。假设有以下边界：

.. code:: python

    hetero_frontier = dgl.heterograph({
        ('user', 'follow', 'user'): ([1, 3, 7], [3, 6, 8]),
        ('user', 'play', 'game'): ([5, 5, 4], [6, 6, 2]),
        ('game', 'played-by', 'user'): ([2], [6])
    }, num_nodes_dict={'user': 10, 'game': 10})

One can also create a block with output nodes User #3, #6, and #8, as
well as Game #2 and #6.

可以创建一个如下的块，块的输出节点为 ``User`` 3、6、8和 ``Game`` 2、6。

.. code:: python

    hetero_block = dgl.to_block(hetero_frontier, {'user': [3, 6, 8], 'block': [2, 6]})

One can also get the input nodes and output nodes by type:

对于这个块，还可以按节点类型获取输入节点和输出节点：

.. code:: python

    # input users and games
    # 输入的User和Game节点
    print(hetero_block.srcnodes['user'].data[dgl.NID], hetero_block.srcnodes['game'].data[dgl.NID])
    # output users and games
    # 输出的User和Game节点
    print(hetero_block.dstnodes['user'].data[dgl.NID], hetero_block.dstnodes['game'].data[dgl.NID])


.. _guide_cn-minibatch-customizing-neighborhood-sampler-impl:

Implementing a Custom Neighbor Sampler

实现一个自定义邻居采样器
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Recall that the following code performs neighbor sampling for node
classification.

回想一下，以下代码在节点分类任务中用于邻居采样。

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

To implement your own neighborhood sampling strategy, you basically
replace the ``sampler`` object with your own. To do that, let’s first
see what :class:`~dgl.dataloading.dataloader.BlockSampler`, the parent class of
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler`, is.

为了实现自定义的邻居采样方法，用户可以将采样方法对象替换为自定义的采样方法对象。
为此，先来看一下
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler`
的父类
:class:`~dgl.dataloading.dataloader.BlockSampler`。

:class:`~dgl.dataloading.dataloader.BlockSampler` is responsible for
generating the list of blocks starting from the last layer, with method
:meth:`~dgl.dataloading.dataloader.BlockSampler.sample_blocks`. The default implementation of
``sample_blocks`` is to iterate backwards, generating the frontiers and
converting them to blocks.

:class:`~dgl.dataloading.dataloader.BlockSampler`
负责使用
:meth:`~dgl.dataloading.dataloader.BlockSampler.sample_blocks`
方法从最后一层开始生成一个列表的块。 ``sample_blocks`` 的默认实现是向后迭代，生成边界并将其转换为块。

Therefore, for neighborhood sampling, **you only need to implement
the**\ :meth:`~dgl.dataloading.dataloader.BlockSampler.sample_frontier`\ **method**. Given which
layer the sampler is generating frontier for, as well as the original
graph and the nodes to compute representations, this method is
responsible for generating a frontier for them.

因此，对于邻居采样，用户仅需要实现**\ :meth:`~dgl.dataloading.dataloader.BlockSampler.sample_frontier`\ **方法**。
给定GNN层、初始图和要计算表示的节点，该方法负责为它们生成边界。

Meanwhile, you also need to pass how many GNN layers you have to the
parent class.

同时，用户还必须将GNN的层数传递给父类。

For example, the implementation of
:class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` can
go as follows.

例如， :class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` 的实现如下。

.. code:: python

    class MultiLayerFullNeighborSampler(dgl.dataloading.BlockSampler):
        def __init__(self, n_layers):
            super().__init__(n_layers)
    
        def sample_frontier(self, block_id, g, seed_nodes):
            frontier = dgl.in_subgraph(g, seed_nodes)
            return frontier

:class:`dgl.dataloading.neighbor.MultiLayerNeighborSampler`, a more
complicated neighbor sampler class that allows you to sample a small
number of neighbors to gather message for each node, goes as follows.

:class:`dgl.dataloading.neighbor.MultiLayerNeighborSampler`
是一个更复杂的邻居采样方法类它允许用户为每个节点采样少量邻居以收集信息，如下所示。

.. code:: python

    class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler):
        def __init__(self, fanouts):
            super().__init__(len(fanouts))
    
            self.fanouts = fanouts
    
        def sample_frontier(self, block_id, g, seed_nodes):
            fanout = self.fanouts[block_id]
            if fanout is None:
                frontier = dgl.in_subgraph(g, seed_nodes)
            else:
                frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
            return frontier

Although the functions above can generate a frontier, any graph that has
the same nodes as the original graph can serve as a frontier.

尽管上面的函数可以生成边界，但是任何拥有与初始图相同节点的图都可用作边界。

For example, if one want to randomly drop inbound edges to the seed
nodes with a probability, one can simply define the sampler as follows:

例如，如果要以某种概率将种子节点的入边随机剔除，则可以按照以下方式简单地定义采样方法：

.. code:: python

    class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
        def __init__(self, p, n_layers):
            super().__init__()
    
            self.n_layers = n_layers
            self.p = p
    
        def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
            # Get all inbound edges to `seed_nodes`
            # 获取种 `seed_nodes` 的所有入边
            src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
            # Randomly select edges with a probability of p
            # 以概率p随机选择边
            mask = torch.zeros_like(src).bernoulli_(self.p)
            src = src[mask]
            dst = dst[mask]
            # Return a new graph with the same nodes as the original graph as a
            # frontier
            # 返回一个与初始图有相同节点的边界
            frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
            return frontier
    
        def __len__(self):
            return self.n_layers

After implementing your sampler, you can create a data loader that takes
in your sampler and it will keep generating lists of blocks while
iterating over the seed nodes as usual.

在实现采样方法后，用户可以创建一个数据加载器，该数据加载器将使用用户自定义的采样方法，
并且它将遍历种子节点生成一系列的块。

.. code:: python

    sampler = MultiLayerDropoutSampler(0.5, 2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)
    
    model = StochasticTwoLayerRGCN(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata     # returns a dict
        output_labels = blocks[-1].dstdata     # returns a dict
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

Heterogeneous Graphs

异构图上自定义采样器
^^^^^^^^^^^^^^^^^^^^

Generating a frontier for a heterogeneous graph is nothing different
than that for a homogeneous graph. Just make the returned graph have the
same nodes as the original graph, and it should work fine. For example,
we can rewrite the ``MultiLayerDropoutSampler`` above to iterate over
all edge types, so that it can work on heterogeneous graphs as well.

为异构图生成边界与为同构图生成边界没有什么不同。只要使返回的图具有与初始图相同的节点，
就可以正常工作。例如，可以重写上面的 ``MultiLayerDropoutSampler`` 以遍历所有边类型，
以便它也可以在异构图上使用。

.. code:: python

    class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
        def __init__(self, p, n_layers):
            super().__init__()
    
            self.n_layers = n_layers
            self.p = p
    
        def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
            # Get all inbound edges to `seed_nodes`
            # 获取 `seed_nodes` 的所有入边
            sg = dgl.in_subgraph(g, seed_nodes)
    
            new_edges_masks = {}
            # Iterate over all edge types
            # 遍历所有边的类型
            for etype in sg.canonical_etypes:
                edge_mask = torch.zeros(sg.number_of_edges(etype))
                edge_mask.bernoulli_(self.p)
                new_edges_masks[etype] = edge_mask.bool()
    
            # Return a new graph with the same nodes as the original graph as a
            # frontier
            # 返回一个与初始图有相同节点的图作为边界
            frontier = dgl.edge_subgraph(new_edge_masks, preserve_nodes=True)
            return frontier
    
        def __len__(self):
            return self.n_layers



