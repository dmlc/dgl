.. _guide_cn-training-edge-classification:

5.2 边分类/回归
---------------------------------------------

:ref:`(English Version) <guide-training-edge-classification>`

Sometimes you wish to predict the attributes on the edges of the graph,
or even whether an edge exists or not between two given nodes. In that
case, you would like to have an *edge classification/regression* model.

有时用户希望预测图中边上的属性值，甚至要预测给定的两个节点之间是否有边。这种情况下，用户需要构建一个边分类/回归模型。

Here we generate a random graph for edge prediction as a demonstration.

以下代码生成了一个随机图用于演示边分类/回归。

.. code:: ipython3

    src = np.random.randint(0, 100, 500)
    dst = np.random.randint(0, 100, 500)
    # make it symmetric
    # 建立对称的边
    edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # synthetic node and edge features, as well as edge labels
    # 建立点和边特征，以及边的标签
    edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
    edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
    edge_pred_graph.edata['label'] = torch.randn(1000)
    # synthetic train-validation-test splits
    # 训练集-验证集-测试集划分
    edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

Overview

概述
~~~~~~~~

From the previous section you have learned how to do node classification
with a multilayer GNN. The same technique can be applied for computing a
hidden representation of any node. The prediction on edges can then be
derived from the representation of their incident nodes.

上一节介绍了如何使用多层GNN进行节点分类。同样的方法可被用于计算任何节点的隐藏表示。
然后就可以从边的两个端点的表示计算得出对边的预测。

The most common case of computing the prediction on an edge is to
express it as a parameterized function of the representation of its
incident nodes, and optionally the features on the edge itself.

在一条边上计算预测值最常见的情况是将预测表示为一个参数化函数，函数的参数是边的两个端点的表示。
参数还可以包括边自身的特征。

Model Implementation Difference from Node Classification

与节点分类在模型实现上的差别
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Assuming that you compute the node representation with the model from
the previous section, you only need to write another component that
computes the edge prediction with the
:meth:`~dgl.DGLHeteroGraph.apply_edges` method.

假设用户使用上一节中用到的模型计算节点表示，用户只需要再编写一个用 :meth:`~dgl.DGLHeteroGraph.apply_edges` 方法计算边预测的组件。

For instance, if you would like to compute a score for each edge for
edge regression, the following code computes the dot product of incident
node representations on each edge.

例如，在边回归任务中，如果用户想为每条边计算一个分数，下面的代码在每一条边上计算了边两端节点隐藏表示的点积。

.. code:: python

    import dgl.function as fn
    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            # h是从5.1节的GNN模型中计算出的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

One can also write a prediction function that predicts a vector for each
edge with an MLP. Such vector can be used in further downstream tasks,
e.g. as logits of a categorical distribution.

用户也可以写一个对每条边通过MLP(多层感知机)预测一个向量的预测函数。
这样的向量可以在下游任务中使用，例如,作为一个未经过归一化的类别分布。

.. code:: python

    class MLPPredictor(nn.Module):
        def __init__(self, in_features, out_classes):
            super().__init__()
            self.W = nn.Linear(in_features * 2, out_classes)
    
        def apply_edges(self, edges):
            h_u = edges.src['h']
            h_v = edges.dst['h']
            score = self.W(torch.cat([h_u, h_v], 1))
            return {'score': score}
    
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            # h是从5.1节的GNN模型中计算出的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

Training loop

训练循环
~~~~~~~~~~~~~

Given the node representation computation model and an edge predictor
model, we can easily write a full-graph training loop where we compute
the prediction on all edges.

给定计算节点表示的模型和边的预测模型后，用户可以轻松地编写在所有边上进行预测的全图训练代码。

The following example takes ``SAGE`` in the previous section as the node
representation computation model and ``DotPredictor`` as an edge
predictor model.

以下代码用上一节的 ``SAGE`` 作为节点表示计算模型， ``DotPredictor`` 作为边预测模型。

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, x):
            h = self.sage(g, x)
            return self.pred(g, h)

In this example, we also assume that the training/validation/test edge
sets are identified by boolean masks on edges. This example also does
not include early stopping and model saving.

在这个例子中，布尔型的掩码区分了训练、验证、测试用的边集合。该例子没有包含早停法和模型保存部分的代码。

.. code:: python

    node_features = edge_pred_graph.ndata['feature']
    edge_label = edge_pred_graph.edata['label']
    train_mask = edge_pred_graph.edata['train_mask']
    model = Model(10, 20, 5)
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(edge_pred_graph, node_features)
        loss = ((pred[train_mask] - edge_label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())

.. _guide_cn-training-edge-classification-heterogeneous-graph:

Heterogeneous graph

异构图上的训练循环
~~~~~~~~~~~~~~~~~~~

Edge classification on heterogeneous graphs is not very different from
that on homogeneous graphs. If you wish to perform edge classification
on one edge type, you only need to compute the node representation for
all node types, and predict on that edge type with
:meth:`~dgl.DGLHeteroGraph.apply_edges` method.

在异构图上进行边预测和在同构图上进行边预测没有太大区别。如果想在某一种边类型上进行边分类任务，
用户只需要计算所有节点类型的节点表示，然后通过 :meth:`~dgl.DGLHeteroGraph.apply_edges` 方法在这种边类型上预测即可。

For example, to make ``DotProductPredictor`` work on one edge type of a
heterogeneous graph, you only need to specify the edge type in
``apply_edges`` method.

例如，为了在异构图的某一类型边上进行 ``DotProductPredictor`` 计算，用户只需要在 ``apply_edges`` 方法中指定边类型即可。

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each edge type computed from
            # the GNN for heterogeneous graphs defined in the node classification
            # section (Section 5.1).
            # h 是从5.1节中对每种类型的边所计算的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

You can similarly write a ``HeteroMLPPredictor``.

同样地，用户可以编写一个 ``HeteroMLPPredictor``。

.. code:: python

    class MLPPredictor(nn.Module):
        def __init__(self, in_features, out_classes):
            super().__init__()
            self.W = nn.Linear(in_features * 2, out_classes)
    
        def apply_edges(self, edges):
            h_u = edges.src['h']
            h_v = edges.dst['h']
            score = self.W(torch.cat([h_u, h_v], 1))
            return {'score': score}
    
        def forward(self, graph, h, etype):
            # h contains the node representations for each edge type computed from
            # the GNN for heterogeneous graphs defined in the node classification
            # section (Section 5.1).
            #h 是从5.1节中对异构图的每种类型的边所计算的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h   #一次性为所有节点类型的 'h'赋值
                graph.apply_edges(self.apply_edges, etype=etype)
                return graph.edges[etype].data['score']

The end-to-end model that predicts a score for each edge on a single
edge type will look like this:

在某一类型的边上为每一条边预测的端到端模型如下所示：

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype)

Using the model simply involves feeding the model a dictionary of node
types and features.

使用模型时只需要简单地向模型提供节点类型和特征的字典。

.. code:: python

    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    label = hetero_graph.edges['click'].data['label']
    train_mask = hetero_graph.edges['click'].data['train_mask']
    node_features = {'user': user_feats, 'item': item_feats}

Then the training loop looks almost the same as that in homogeneous
graph. For instance, if you wish to predict the edge labels on edge type
``click``, then you can simply do

然后训练的循环部分就和同构图的循环基本一致了。例如，如果用户想预测边类型为 ``click`` 的边的标签，只需要按下例编写代码。

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(hetero_graph, node_features, 'click')
        loss = ((pred[train_mask] - label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


Predicting Edge Type of an Existing Edge on a Heterogeneous Graph

在异构图中预测图中已经存在的边的边类型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes you may want to predict which type an existing edge belongs
to.

有时候用户可能想预测图中已经存在的边属于哪个边类型。

For instance, given the
:ref:`heterogeneous graph example <guide-training-heterogeneous-graph-example>`,
your task is given an edge connecting a user and an item, to predict whether
the user would ``click`` or ``dislike`` an item.

例如，给定 :ref:`heterogeneous graph example <guide-training-heterogeneous-graph-example>`
所述的异构图，用户的任务是给定一条连接user和item的边，预测 ``user`` 和 ``item``
之间的连接边类型是 ``click`` 还是 ``dislike``。

This is a simplified version of rating prediction, which is common in
recommendation literature.

这是评分预测的一个简化版本，在推荐场景中很常见。

You can use a heterogeneous graph convolution network to obtain the node
representations. For instance, you can still use the
:ref:`RGCN defined previously <guide-training-rgcn-node-classification>`
for this purpose.

用户可以使用一个异构图卷积网络来获取节点表示。例如，用户仍然可以将 :ref:`前述的RGCN <guide_cn-training-rgcn-node-classification>`
用于此目的。

To predict the type of an edge, you can simply repurpose the
``HeteroDotProductPredictor`` above so that it takes in another graph
with only one edge type that “merges” all the edge types to be
predicted, and emits the score of each type for every edge.

要预测一条边的类型，用户可以简单地更换上述提到的 ``HeteroDotProductPredictor`` 的用途，
给 ``HeteroDotProductPredictor`` 输入另外一个将所有要预测的边类型合并了成一个边类型的图，
并为每条边计算出每种边类型的可能得分。

In the example here, you will need a graph that has two node types
``user`` and ``item``, and one single edge type that “merges” all the
edge types from ``user`` and ``item``, i.e. ``click`` and ``dislike``.
This can be conveniently created using the following syntax:

下面的例子中，用户需要一个拥有 ``user`` 和 ``item`` 两个节点类型和一个边类型的图。
该边类型通过合并所有从 ``user`` 到 ``item`` 的边类型（例如 ``like`` 和 ``dislike``）而来。
用户可以很方便地用关系切片的方式创建这个图。

.. code:: python

    dec_graph = hetero_graph['user', :, 'item']

which returns a heterogeneous graphs with node type ``user`` and ``item``,
as well as a single edge type combining all edge types in between, i.e.
``click`` and ``dislike``.

这个方法会返回一个异构图，它具有 ``user`` 和 ``item`` 两种节点类型，
以及把它们之间的所有边的类型(如，``click`` 和 ``dislike``)进行合并后的单一边类型。

Since the statement above also returns the original edge types as a
feature named ``dgl.ETYPE``, we can use that as labels.

由于上面这行代码将原来的边类型存成边特征 ``dgl.ETYPE``，用户可以将它作为标签使用。

.. code:: python

    edge_label = dec_graph.edata[dgl.ETYPE]

Given the graph above as input to the edge type predictor module, you
can write your predictor module as follows.

将上述图作为边类型预测模块的输入，用户可以按如下方式编写预测模块：

.. code:: python

    class HeteroMLPPredictor(nn.Module):
        def __init__(self, in_dims, n_classes):
            super().__init__()
            self.W = nn.Linear(in_dims * 2, n_classes)
    
        def apply_edges(self, edges):
            x = torch.cat([edges.src['h'], edges.dst['h']], 1)
            y = self.W(x)
            return {'score': y}
    
        def forward(self, graph, h):
            # h contains the node representations for each edge type computed from
            # the GNN for heterogeneous graphs defined in the node classification
            # section (Section 5.1).
            # h 是从5.1节中对异构图的每种类型的边所计算的节点表示
            with graph.local_scope():
                graph.ndata['h'] = h   #一次性为所有节点类型的 'h'赋值
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

The model that combines the node representation module and the edge type
predictor module is the following:

结合了节点表示模块和边类型预测模块的模型如下所示：

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroMLPPredictor(out_features, len(rel_names))
        def forward(self, g, x, dec_graph):
            h = self.sage(g, x)
            return self.pred(dec_graph, h)

The training loop then simply be the following:

训练的循环部分如下所示：

.. code:: python

    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    node_features = {'user': user_feats, 'item': item_feats}
    
    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        logits = model(hetero_graph, node_features, dec_graph)
        loss = F.cross_entropy(logits, edge_label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


DGL provides `Graph Convolutional Matrix
Completion <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
as an example of rating prediction, which is formulated by predicting
the type of an existing edge on a heterogeneous graph. The node
representation module in the `model implementation
file <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
is called ``GCMCLayer``. The edge type predictor module is called
``BiDecoder``. Both of them are more complicated than the setting
described here.

DGL提供了`Graph Convolutional Matrix
Completion <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
作为打分预测的示例，它是为了预测异构图中已经存在的边的边类型任务准备的。
`模型实现文件中 <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__
的节点表示模块称作 ``GCMCLayer``。边类型预测模块称作 ``BiDecoder``。这两个模块都比前述的示例代码要复杂一些。

