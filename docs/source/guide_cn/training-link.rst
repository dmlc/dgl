.. _guide_cn-training-link-prediction:

5.3 链接预测
---------------------------

:ref:`(English Version) <guide-training-link-prediction>`

In some other settings you may want to predict whether an edge exists
between two given nodes or not. Such model is called a *link prediction*
model.

在某些场景中，用户可能希望预测给定节点之间是否存在边，这样的模型称作 **链接预测** 模型。

Overview

概述
~~~~~~~~

A GNN-based link prediction model represents the likelihood of
connectivity between two nodes :math:`u` and :math:`v` as a function of
:math:`\boldsymbol{h}_u^{(L)}` and :math:`\boldsymbol{h}_v^{(L)}`, their
node representation computed from the multi-layer GNN.

基于GNN的链接预测模型基于 :math:`u`、:math:`v` 两个节点的表示
:math:`\boldsymbol{h}_u^{(L)}` 和  :math:`\boldsymbol{h}_v^{(L)}` 来预测它们之间连接的可能性，
其中  :math:`\boldsymbol{h}_u^{(L)}` 和  :math:`\boldsymbol{h}_v^{(L)}` 由多层GNN计算得出。

.. math::

   y_{u,v} = \phi(\boldsymbol{h}_u^{(L)}, \boldsymbol{h}_v^{(L)})

In this section we refer to :math:`y_{u,v}` the *score* between node
:math:`u` and node :math:`v`.

本节将节点 :math:`u` 和 :math:`v` 之间连接可能性的 *得分* 记作 :math:`y_{u,v}`。

Training a link prediction model involves comparing the scores between
nodes connected by an edge against the scores between an arbitrary pair
of nodes. For example, given an edge connecting :math:`u` and :math:`v`,
we encourage the score between node :math:`u` and :math:`v` to be higher
than the score between node :math:`u` and a sampled node :math:`v'` from
an arbitrary *noise* distribution :math:`v' \sim P_n(v)`. Such
methodology is called *negative sampling*.

训练一个链接预测模型涉及到比较两个相连接节点之间的得分与任意一对节点之间的得分。
例如，给定一条连接 :math:`u` 和 :math:`v` 的边，一个好的模型希望 :math:`u` 和 :math:`v` 之间的得分要高于
:math:`u` 和从一个任意的噪声分布 :math:`v′∼Pn(v)` 中所采样的节点 :math:`v′` 之间的得分。这样的方法称作 *负采样*。

There are lots of loss functions that can achieve the behavior above if
minimized. A non-exhaustive list include:

许多损失函数都可以达到上述目标，例子包括但不限于。

-  交叉熵损失:
   :math:`\mathcal{L} = - \log \sigma (y_{u,v}) - \sum_{v_i \sim P_n(v), i=1,\dots,k}\log \left[ 1 - \sigma (y_{u,v_i})\right]`
-  贝叶斯个性化排序损失:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} - \log \sigma (y_{u,v} - y_{u,v_i})`
-  间隔损失:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} \max(0, M - y_{u, v} + y_{u, v_i})`,
   where :math:`M` is a constant hyperparameter.
   其中 :math:`M` 是常数项超参数。

You may find this idea familiar if you know what `implicit
feedback <https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf>`__ or
`noise-contrastive
estimation <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`__
is.

如果用户熟悉 `implicit feedback <https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf>`__ 和
`noise-contrastive estimation <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`__ ，
可能会发现这些想法都很类似。

Model Implementation Difference from Edge Classification

与边分类模型的实现区别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The neural network model to compute the score between :math:`u` and
:math:`v` is identical to the edge regression model described
:ref:`above <guide-training-edge-classification>`.

计算 :math:`u` 和 :math:`v` 之间分数的神经网络模型与上述 :ref:`<guide_cn-training-edge-classification>` 中所述的边回归模型相同。

Here is an example of using dot product to compute the scores on edges.

下面是使用点积计算边得分的例子。

.. code:: python

    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

Training loop
训练循环
~~~~~~~~~~~~~

Because our score prediction model operates on graphs, we need to
express the negative examples as another graph. The graph will contain
all negative node pairs as edges.

因为上述的评分预测模型在图上进行计算，用户需要将负采样的样本表示为另外一个图，
图中包含所有负采样的节点对作为边。

The following shows an example of expressing negative examples as a
graph. Each edge :math:`(u,v)` gets :math:`k` negative examples
:math:`(u,v_i)` where :math:`v_i` is sampled from a uniform
distribution.

下面的例子展示了将负采样的样本表示为一个图。每一条边 :math:`(u,v)` 都有 :math:`k`
个对应的负采样样本 :math:`(u,v_i)`，其中 :math:`v_i` 是从均匀分布中采样的。

.. code:: python

    def construct_negative_graph(graph, k):
        src, dst = graph.edges()
    
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,))
        return dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())

The model that predicts edge scores is the same as that of edge
classification/regression.

预测边得分的模型和边分类/回归模型中的预测边得分模型相同。

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, neg_g, x):
            h = self.sage(g, x)
            return self.pred(g, h), self.pred(neg_g, h)

The training loop then repeatedly constructs the negative graph and
computes loss.

训练的循环部分重复构建负采样图并计算损失函数值。

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()
    
    node_features = graph.ndata['feat']
    n_features = node_features.shape[1]
    k = 5
    model = Model(n_features, 100, 100)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        negative_graph = construct_negative_graph(graph, k)
        pos_score, neg_score = model(graph, negative_graph, node_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


After training, the node representation can be obtained via

训练后，节点表示可以通过以下代码获取。

.. code:: python

    node_embeddings = model.sage(graph, node_features)

There are multiple ways of using the node embeddings. Examples include
training downstream classifiers, or doing nearest neighbor search or
maximum inner product search for relevant entity recommendation.

(实际应用中)，有着许多使用节点嵌入的方法。例如训练下游的分类器，或为相关实体推荐进行最近邻搜索或最大内积搜索。

Heterogeneous graphs
异构图上的训练循环
~~~~~~~~~~~~~~~~~~~~

Link prediction on heterogeneous graphs is not very different from that
on homogeneous graphs. The following assumes that we are predicting on
one edge type, and it is easy to extend it to multiple edge types.

异构图上的链接预测和同构图上的链接预测没有太大区别。以下假设在一种边类型上进行预测，
用户可以很容易地将其拓展为对多种边类型上进行预测。

For example, you can reuse the ``HeteroDotProductPredictor``
:ref:`above <guide-training-edge-classification-heterogeneous-graph>`
for computing the scores of the edges of an edge type for link prediction.

例如，用户可以重复使用
:ref:`上述 <guide_cn-training-edge-classification-heterogeneous-graph>`
的 ``HeteroDotProductPredictor`` 为某一种边类型计算边上的连接可能性得分。

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each node type computed from
            # the GNN defined in the previous section (Section 5.1).
            # h 是从5.1节中对异构图的每种类型的边所计算的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

To perform negative sampling, one can construct a negative graph for the
edge type you are performing link prediction on as well.

要执行负采样，用户可以对要进行链接预测的边类型构造一个负采样图。

.. code:: python

    def construct_negative_graph(graph, k, etype):
        utype, _, vtype = etype
        src, dst = graph.edges(etype=etype)
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.number_of_nodes(vtype), (len(src) * k,))
        return dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: graph.number_of_nodes(ntype) for ntype in graph.ntypes})

The model is a bit different from that in edge classification on
heterogeneous graphs since you need to specify edge type where you
perform link prediction.

该模型与异构图上边分类的模型有些不同，因为用户需要指定在哪种边类型上进行连接预测。

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, neg_g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype), self.pred(neg_g, h, etype)

The training loop is similar to that of homogeneous graphs.

训练的循环部分和同构图时一致。

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        # 间隔损失
        n_edges = pos_score.shape[0]
        return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()
    
    k = 5
    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        negative_graph = construct_negative_graph(hetero_graph, k, ('user', 'click', 'item'))
        pos_score, neg_score = model(hetero_graph, negative_graph, node_features, ('user', 'click', 'item'))
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())



