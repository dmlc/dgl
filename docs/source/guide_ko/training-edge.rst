.. _guide_ko-training-edge-classification:

5.2 에지 분류 및 리그레션(Regression)
--------------------------------

:ref:`(English Version) <guide-training-edge-classification>`

때론 그래프의 에지들의 속성을 예측을 원하는 경우가 있다. 이를 위해서 *에지 분류/리그레션* 모델을 만들고자 한다.

우선, 예제로 사용할 에지 예측을 위한 임의의 그래프를 만든다.

.. code:: python

    src = np.random.randint(0, 100, 500)
    dst = np.random.randint(0, 100, 500)
    # make it symmetric
    edge_pred_graph = dgl.graph((np.concatenate([src, dst]), np.concatenate([dst, src])))
    # synthetic node and edge features, as well as edge labels
    edge_pred_graph.ndata['feature'] = torch.randn(100, 10)
    edge_pred_graph.edata['feature'] = torch.randn(1000, 10)
    edge_pred_graph.edata['label'] = torch.randn(1000)
    # synthetic train-validation-test splits
    edge_pred_graph.edata['train_mask'] = torch.zeros(1000, dtype=torch.bool).bernoulli(0.6)

개요
~~~~~~~~~

앞 절에서 우리는 멀티 레이어 GNN을 사용해서 노드 분류하는 방법을 알아봤다. 임의의 노드에 대한 hidden representation을 계산하기 위해서 같은 기법을 적용한다. 그러면 에지들에 대한 예측은 그것들의 부속 노드들의 representation들로 부터 도출할 수 있다.

에지에 대한 예측을 계산하는 가장 일반적인 방법은 그 에지의 부속 노드들의 representation들과 부수적으로 그 에지에 대한 피쳐들의 parameterized 함수로 표현하는 것이다.

노드 분류 모델과 구현상의 차이점
~~~~~~~~~~~~~~~~~~~~~~~~

이전 절에서 만든 모델을 사용해서 노드 representation을 계산한다고 가정하면, :meth:`~dgl.DGLGraph.apply_edges` 메소드로 에지 예측을 계산하는 컴포넌트만 작성하면 된다.

예를 들어, 에지 리그레션을 위해서 각 에지에 대한 점수를 계산하고자 한다면, 아래 코드와 같이 각 에지에 대한 부속 노드의 representation들의 dot product를 계산하면 된다.

.. code:: python

    import dgl.function as fn
    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

또한 MLP를 사용해서 각 에지에 대한 벡터 값을 예측하는 예측하는 함수를 작성할 수도 있다. 이 벡터 값은 미래의 다운스트림 테스크들에 사용될 수 있다. 즉, 범주형 분류의 logit으로 사용.

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
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

학습 룹(loop)
~~~~~~~~~~~

노드 representation 계산 모델과 에지 예측 모델을 만들었다면, 모든 에지들에 대한 예측값을 계산하는 전체 그래프를 이용한 학습 룹을 작성할 수 있다.

노드 representation 계산 모델로 ``SAGE`` 를, 에지 예측 모델로 ``DotPredictor`` 을 사용한다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, x):
            h = self.sage(g, x)
            return self.pred(g, h)

이 예제에서 학습/검증/테스트 에지 셋이 에지의 이진 마스크로 구분된다고 가정한다. 또한 early stopping이나 모델 저장은 포함하지 않는다.

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

.. _guide_ko-training-edge-classification-heterogeneous-graph:

Heterogeneous 그래프
~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프들에 대한 에지 분류는 homogeneous 그래프와 크게 다르지 않다. 하나의 에지 타입에 대해서 에지 분류를 수행하자 한다면, 모든 노드 티압에 대한 노드 representation을 구하고, :meth:`~dgl.DGLGraph.apply_edges` 메소드를 사용해서 에지 타입을 예측하면 된다.

예를 들면, heterogeneous 그래프의 하나의 에지 타입에 대한 동작하는 ``DotProductPredictor`` 를 작성하고자 한다면, ``apply_edges`` 메소드에 해당 에지 타입을 명시하기만 하면 된다.

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each edge type computed from
            # the GNN for heterogeneous graphs defined in the node classification
            # section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

비슷하게 ``HeteroMLPPredictor`` 를 작성할 수 있다.

.. code:: python

    class HeteroMLPPredictor(nn.Module):
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
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(self.apply_edges, etype=etype)
                return graph.edges[etype].data['score']

특정 타입의 에지에 대해서, 각 에지의 점수를 예측하는 end-to-end 모델을 다음과 같다:

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype)

모델을 사용하는 방법은 노드 타입과 피쳐들에 대한 사전을 모델에 간단하게 입력하면 된다.

.. code:: python

    model = Model(10, 20, 5, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    label = hetero_graph.edges['click'].data['label']
    train_mask = hetero_graph.edges['click'].data['train_mask']
    node_features = {'user': user_feats, 'item': item_feats}

학습 룹은 homogeneous 그래프의 것과 거의 유사하다. 예를 들어, 에지 타입 ``click`` 에 대한 에지 레이블을 예측하는 것은 다음과 같이 간단히 구현된다.

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    for epoch in range(10):
        pred = model(hetero_graph, node_features, 'click')
        loss = ((pred[train_mask] - label[train_mask]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())


Heterogeneous 그래프의 에지들에 대한 에지 타입 예측하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

주어진 에지의 타입을 예측하는 일도 종종 하게된다.

:ref:`heterogeneous 그래프 예제 <guide_ko-training-heterogeneous-graph-example>` 에서는 user와 item을 연결하는 에지가 주어졌을 때, user가 ``click`` 을 선택할지, ``dislike`` 를 선택할지를 예측하고 있다.

이는 추천에서 흔히 쓰이는 평가 예측의 간략한 버전이다.

노드 representation을 얻기 위해서 heterogeneous graph convolution 네트워크를 사용할 수 있다. 이를 위해서 :ref:`이전에 정의한 RGCN <guide_ko-training-rgcn-node-classification>` 를 사용하는 것도 가능하다.

에지 타입을 예측하기 위해서 ``HeteroDotProductPredictor`` 의 용도를 간단히 변경해서 예측할 모든 에지 타입을 “병합“하고 모든 에지들의 각 타입에 대한 점수를 내보내는 하나의 에지 타입만 있는 다른 그래프를 취하게하면 된다.

이 예제에 적용해보면, ``user`` 와 ``item`` 두 노트 타입을 갖으며 ``user`` 와 ``item`` 에 대한 ``click`` 이나 ``dislike`` 같은 모든 에지 타입을 병합하는 단일 에지 타입을 갖는 그래프가 필요하다. 다음 문장으로 간단하게 생성할 수 있다.

.. code:: python

    dec_graph = hetero_graph['user', :, 'item']

이 함수는 ``user`` 와 ``item`` 을 노드 타입으로 갖고, 두 노드 타입을 연결하고 있는 모든 에지 타입(예, ``click`` 와 ``dislike`` )을 합친 단일 에지 타입을 갖는 heterogeneous 그래프를 리턴한다.

위 코드는 원래의 에지 타입을 ``dgl.ETYPE`` 이라는 이름의 피처로 리턴하기 때문에, 이를 레이블로 사용할 수 있다.

.. code:: python

    edge_label = dec_graph.edata[dgl.ETYPE]

에지 타입 예측 모듈의 입력으로 위 그래프를 사용해서 예측 모델을 다음과 같이 작성한다.

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
            with graph.local_scope():
                graph.ndata['h'] = h   # assigns 'h' of all node types in one shot
                graph.apply_edges(self.apply_edges)
                return graph.edata['score']

노드 representation 모듈과 에지 타입 예측 모듈을 합친 모델은 다음과 같다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroMLPPredictor(out_features, len(rel_names))
        def forward(self, g, x, dec_graph):
            h = self.sage(g, x)
            return self.pred(dec_graph, h)

학습 룹은 아래와 같이 간단하다.

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

DGL은 heterogeneous 그래프의 에지들에 대한 타입을 예측하는 문제인 평가 예측 예제로 `Graph Convolutional Matrix Completion <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__ 를 제공한다. `모델 구현 파일 <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__ 에 있는 노드 representation 모듈은 ``GCMCLayer`` 라고 불린다. 이 둘은 여기서 설명하기에는 너무 복잡하니 자세한 설명은 생략한다.
