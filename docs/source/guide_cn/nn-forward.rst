.. _guide_cn-nn-forward:

3.2 编写DGL NN模块的forward函数
-----------------------------

In NN module, ``forward()`` function does the actual message passing and
computation. Compared with PyTorch’s NN module which usually takes
tensors as the parameters, DGL NN module takes an additional parameter
:class:`dgl.DGLGraph`. The
workload for ``forward()`` function can be split into three parts:

在NN模块中，forward()函数执行了实际的消息传递和计算。与通常以张量为参数的PyTorch NN模块相比，
DGL NN模块采用了1个附加参数dgl.DGLGraph。forward()函数的工作可以分为3个部分：

-  Graph checking and graph type specification.

-  Message passing.

-  Feature update.

● 图检验和图类型规范。
● 消息传递和聚合。
● 聚合后，更新特征作为输出。

The rest of the section takes a deep dive into the ``forward()`` function in SAGEConv example.

下面展示了SAGEConv示例中的forward()函数。

Graph checking and graph type specification

图检验和图类型规范
~~~~~~~~~~~~~~~

.. code::

        def forward(self, graph, feat):
            with graph.local_scope():
                # Specify graph type then expand input feature according to graph type
                # 指定图类型，然后根据图类型扩展输入特征
                feat_src, feat_dst = expand_as_pair(feat, graph)

``forward()`` needs to handle many corner cases on the input that can
lead to invalid values in computing and message passing. One typical check in conv modules
like :class:`~dgl.nn.pytorch.conv.GraphConv` is to verify that the input graph has no 0-in-degree nodes.
When a node has 0 in-degree, the ``mailbox`` will be empty and the reduce function will produce
all-zero values. This may cause silent regression in model performance. However, in
:class:`~dgl.nn.pytorch.conv.SAGEConv` module, the aggregated representation will be concatenated
with the original node feature, the output of ``forward()`` will not be all-zero. No such check is
needed in this case.

forward()函数需要处理输入上的许多极端情况，这些情况可能导致计算和消息传递中的值无效。在GraphConv等conv模块中，
一个典型的检验方法是验证输入图中没有入度为0的节点。当1个节点入度为0时，mailbox将为空，并且聚合函数的输出值全为0。
这可能会导致模型性能的无声回归。但是，在SAGEConv模块中，被聚合的表示将与初始节点特征连接起来，forward()函数的输出不会全为0。
在这种情况下，无需进行此类检验。

DGL NN module should be reusable across different types of graph input
including: homogeneous graph, heterogeneous
graph (:ref:`guide-graph-heterogeneous`), subgraph
block (:ref:`guide-minibatch`).

DGL NN模块可在不同类型的图输入中重复使用，包括：同构图、异构图（1.5 异构图）和子图区块（第6章：在大图上的随机训练）。

The math formulas for SAGEConv are:

SAGEConv的数学公式为：

.. math::


   h_{\mathcal{N}(dst)}^{(l+1)}  = \mathrm{aggregate}
           \left(\{h_{src}^{l}, \forall src \in \mathcal{N}(dst) \}\right)

.. math::

    h_{dst}^{(l+1)} = \sigma \left(W \cdot \mathrm{concat}
           (h_{dst}^{l}, h_{\mathcal{N}(dst)}^{l+1} + b) \right)

.. math::

    h_{dst}^{(l+1)} = \mathrm{norm}(h_{dst}^{l})

One needs to specify the source node feature ``feat_src`` and destination
node feature ``feat_dst`` according to the graph type.
:meth:``~dgl.utils.expand_as_pair`` is a function that specifies the graph
type and expand ``feat`` into ``feat_src`` and ``feat_dst``.
The detail of this function is shown below.

源节点特征feat_src和目标节点特征feat_dst需要根据图类型被指定。
用于指定图类型并将feat扩展为feat_src和feat_dst的函数为expand_as_pair()。
该函数的细节如下所示。

.. code::

    def expand_as_pair(input_, g=None):
        if isinstance(input_, tuple):
            # Bipartite graph case
            # 二部图的情况
            return input_
        elif g is not None and g.is_block:
            # Subgraph block case
            # 子图块的情况
            if isinstance(input_, Mapping):
                input_dst = {
                    k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                    for k, v in input_.items()}
            else:
                input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
            return input_, input_dst
        else:
            # Homogeneous graph case
            # 同构图的情况
            return input_, input_

For homogeneous whole graph training, source nodes and destination nodes
are the same. They are all the nodes in the graph.

对于同构全图训练，源节点和目标节点相同。它们都是图中的所有节点。

For heterogeneous case, the graph can be split into several bipartite
graphs, one for each relation. The relations are represented as
``(src_type, edge_type, dst_dtype)``. When it identifies that the input feature
``feat`` is a tuple, it will treat the graph as bipartite. The first
element in the tuple will be the source node feature and the second
element will be the destination node feature.

在异构的情况下，图可以分为几个二部图，每种关系对应1个。关系表示为(src_type, edge_type, dst_dtype)。
当输入特征feat是1个元组时，图将会被视为二部图。元组中的第1个元素为源节点特征，第2个元素为目标节点特征。

In mini-batch training, the computing is applied on a subgraph sampled
based on a bunch of destination nodes. The subgraph is called as
``block`` in DGL. After message passing, only those destination nodes
will be updated since they have the same neighborhood as the one they
have in the original full graph. In the block creation phase,
``dst nodes`` are in the front of the node list. One can find the
``feat_dst`` by the index ``[0:g.number_of_dst_nodes()]``.

在小批次训练中，计算应用于给定的一堆目标节点所采样的子图。子图在DGL中称为block。
消息传递后，由于那些目标节点拥有和初始完整图中相同的邻域，因此仅更新这些目标节点。
在区块创建的阶段，dst nodes位于节点列表的最前面。通过索引[0:g.number_of_dst_nodes()]可以找到feat_dst。

After determining ``feat_src`` and ``feat_dst``, the computing for the
above three graph types are the same.

确定feat_src和feat_dst之后，以上3种图类型的计算相同。

Message passing and reducing

消息传递和聚合
~~~~~~~~~~~~

.. code::

                import dgl.function as fn
                import torch.nn.functional as F
                from dgl.utils import check_eq_shape

                if self._aggre_type == 'mean':
                    graph.srcdata['h'] = feat_src
                    graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                elif self._aggre_type == 'gcn':
                    check_eq_shape(feat)
                    graph.srcdata['h'] = feat_src
                    graph.dstdata['h'] = feat_dst     # same as above if homogeneous # 在同构图的情况下，和上述相同
                    graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
                    # divide in_degrees
                    # 除以入度
                    degs = graph.in_degrees().to(feat_dst)
                    h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
                elif self._aggre_type == 'max_pool':
                    graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
                    graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
                    h_neigh = graph.dstdata['neigh']
                else:
                    raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

                # GraphSAGE GCN does not require fc_self.
                # GraphSAGE图卷积网络不需要
                if self._aggre_type == 'gcn':
                    rst = self.fc_neigh(h_neigh)
                else:
                    rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)

The code actually does message passing and reducing computing. This part
of code varies module by module. Note that all the message passing in
the above code are implemented using :meth:`~dgl.DGLGraph.update_all` API and
``built-in`` message/reduce functions to fully utilize DGL’s performance
optimization as described in :ref:`guide-message-passing-efficient`.

该代码实际上执行消息传递和聚合的计算。这部分代码因模块而异。
请注意，以上代码中的所有消息传递均使用update_all()API和built-in消息/聚合函数来实现，
以充分利用DGL的性能优化，如第2章：消息传递中所述。

Update feature after reducing for output

聚合后，更新特征作为输出
~~~~~~~~~~~~~~~~~~~~

.. code::

                # activation
                # 激活
                if self.activation is not None:
                    rst = self.activation(rst)
                # normalization
                if self.norm is not None:
                    rst = self.norm(rst)
                return rst

The last part of ``forward()`` function is to update the feature after
the ``reduce function``. Common update operations are applying
activation function and normalization according to the option set in the
object construction phase.

forward()函数的最后一部分是在reduce function后更新特征。
常见的更新操作是根据对象构造阶段中设置的选项应用激活函数和归一化。
