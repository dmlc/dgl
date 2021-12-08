.. _guide_cn-training-eweight:

5.5 使用边权重
----------------------------------

:ref:`(English Version) <guide-training-eweight>`

在一个加权图里，每条边都有一个有意义的标量权重。例如，边权重可以是连接强度或者信心指数。
人们自然会想要在模型开发中使用它们。

使用边权重的消息传递
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

大部分图神经网络在前馈计算中仅通过消息传递引入图结构信息。一个消息传递运算可以视为一个函数。
这个函数的输入变量是一个邻接矩阵和其他输入特征。对于一个不带权重的图，邻接矩阵里的元素不是零就是一。
值为一的元素表示一条边。对于一个加权图，非零的元素可以取任意标量值。这等价于把每条消息和对应的边权重相乘，
即`图注意力网络 <https://arxiv.org/pdf/1710.10903.pdf>`__中的做法。

在DGL里可以通过以下步骤实现这一需求：

- 把边权重保存为一个边特征
- 在消息函数里，用保存的边特征与对应边的原始消息相乘

考虑以下基于DGL的消息传递示例：

.. code::

    import dgl.function as fn

    # 假定graph.ndata['ft']存储了输入节点特征
    graph.update_all(fn.copy_u('ft', 'm'), fn.sum('m', 'ft'))

可以将其按以下方式修改以支持边权重：

.. code::

    import dgl.function as fn

    # 将边权重保存为一个边特征。边权重是一个形状为(E, *)的张量。
    # E是边的数量
    graph.edata['w'] = eweight

    # 假定graph.ndata['ft']存储了输入节点特征
    graph.update_all(fn.u_mul_e('ft', 'w', 'm'), fn.sum('m', 'ft'))

在NN模块中使用边权重
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

用户可以通过修改NN模块中所有的消息传递操作来给NN模块增加边权重支持。以下代码块提供了一个例子。

.. code::
    import dgl.function as fn
    import torch.nn as nn

    class GNN(nn.Module):
        def __init__(self, in_feats, out_feats):
            super().__init__()
            self.linear = nn.Linear(in_feats, out_feats)

        def forward(self, g, feat, edge_weight=None):
            with g.local_scope():
                g.ndata['ft'] = self.linear(feat)
                if edge_weight is None:
                    msg_func = fn.copy_u('ft', 'm')
                else:
                    g.edata['w'] = edge_weight
                    msg_func = fn.u_mul_e('ft', 'w', 'm')
                g.update_all(msg_func, fn.sum('m', 'ft'))
                return g.ndata['ft']

DGL内置的NN模块如果在forward函数中支持一个可选的:attr:`edge_weight`变量，那么它们已经支持了边权重。

用户可能会需要标准化原始边权重。DGL提供了一个满足这个功能的函数
:func:`~dgl.nn.pytorch.conv.EdgeWeightNorm`。
