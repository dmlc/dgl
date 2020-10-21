.. _guide_cn-nn-construction:

3.1 DGL的NN模块构建函数
-----------------------------

:ref:`(English Version) <guide-nn-construction>`

The construction function performs the following steps:

构建函数会做以下几点：

1. Set options.
2. Register learnable parameters or submodules.
3. Reset parameters.

1. 设置选项。
2. 注册可学习的参数或者子模块。
3. 重置参数。

.. code::

    import torch.nn as nn

    from dgl.utils import expand_as_pair

    class SAGEConv(nn.Module):
        def __init__(self,
                     in_feats,
                     out_feats,
                     aggregator_type,
                     bias=True,
                     norm=None,
                     activation=None):
            super(SAGEConv, self).__init__()

            self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
            self._out_feats = out_feats
            self._aggre_type = aggregator_type
            self.norm = norm
            self.activation = activation

In construction function, one first needs to set the data dimensions. For
general PyTorch module, the dimensions are usually input dimension,
output dimension and hidden dimensions. For graph neural, the input
dimension can be split into source node dimension and destination node
dimension.

在构造函数中，用户首先需要设置数据的维度。对于一般的PyTorch模块，维度通常包括输入的维度、输出维度和隐层的维度。
对于图神经网络，输入维度可被分为源节点特征维度和目标节点的特征维度。

Besides data dimensions, a typical option for graph neural network is
aggregation type (``self._aggre_type``). Aggregation type determines how
messages on different edges are aggregated for a certain destination
node. Commonly used aggregation types include ``mean``, ``sum``,
``max``, ``min``. Some modules may apply more complicated aggregation
like an ``lstm``.

除了数据维度，图神经网络的一个典型选项是聚合类型(``self._aggre_type``)。对于特定目标节点，聚合类型决定了如何聚合不同边上的信息。
常用的聚合类型包括 ``mean``、 ``sum``、 ``max``、 ``min``。一些模块可能会使用更加复杂的聚合函数，比如一个 ``lstm``。

``norm`` here is a callable function for feature normalization. In the
SAGEConv paper, such normalization can be l2 normalization:
:math:`h_v = h_v / \lVert h_v \rVert_2`.

上面代码里的 ``norm`` 是用于特征归一化的可调用函数。在SAGEConv论文里，归一化可以是L2标准化:
:math:`h_v = h_v / \lVert h_v \rVert_2`。

.. code::

            # aggregator type: mean, max_pool, lstm, gcn
            # 聚合类型：mean、max_pool、lstm、gcn
            if aggregator_type not in ['mean', 'max_pool', 'lstm', 'gcn']:
                raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
            if aggregator_type == 'max_pool':
                self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
            if aggregator_type == 'lstm':
                self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
            if aggregator_type in ['mean', 'max_pool', 'lstm']:
                self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
            self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
            self.reset_parameters()

Register parameters and submodules. In SAGEConv, submodules vary
according to the aggregation type. Those modules are pure PyTorch nn
modules like ``nn.Linear``, ``nn.LSTM``, etc. At the end of construction
function, weight initialization is applied by calling
``reset_parameters()``.

注册参数和子模块。在SAGEConv中，子模块根据聚合类型而有所不同。这些模块是纯PyTorch nn模块，例如 ``nn.Linear``、 ``nn.LSTM`` 等。
构造函数的最后调用了 ``reset_parameters()`` 进行权重初始化。

.. code::

        def reset_parameters(self):
            """Reinitialize learnable parameters."""
            """重新初始化可学习的参数。"""
            gain = nn.init.calculate_gain('relu')
            if self._aggre_type == 'max_pool':
                nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
            if self._aggre_type == 'lstm':
                self.lstm.reset_parameters()
            if self._aggre_type != 'gcn':
                nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
            nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
