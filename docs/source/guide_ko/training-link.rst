.. _guide_ko-training-link-prediction:

5.3 링크 예측
-----------

:ref:`(English Version) <guide-training-link-prediction>`

어떤 두 노드들 사이에 에지가 존재하는지 아닌지를 예측하고 싶은 경우가 있고, 이를 *링크 예측 과제* 라고 한다.

개요
~~~~~~~~~

GNN 기반의 링크 예측 모델은 두 노드 :math:`u` 와 :math:`v` 간의 연결 가능도(likelihood)를 :math:`\boldsymbol{h}_u^{(L)}` 의 함수로 표현하는데, 여기서 :math:`\boldsymbol{h}_v^{(L)}` 는 멀티-레이어 GNN을 통해서 계단된 노드 representation이다. 

.. math::


   y_{u,v} = \phi(\boldsymbol{h}_u^{(L)}, \boldsymbol{h}_v^{(L)})

:math:`y_{u,v}` 는 노드 :math:`u` 와 :math:`v` 사이의 점수를 뜻 한다.

링크 예측 모델을 학습시키는 것은 에지로 연결된 두 노드들에 대한 점수와 임의의 두 노드 쌍에 대한 점수를 비교하면서 이뤄진다. 예를 들어, 노드 :math:`u` 와 :math:`v` 사이에 에지가 존재하는 경우 노드 :math:`u` 와 :math:`v` 사이의 점수가 노드 :math:`u` 와 임의의 *노이즈* 분표 :math:`v' \sim P_n(v)`에 따라 샘플링된 노드 :math:`v'` 간의 점수보다 높도록 하는 학습이다.

위를 달성하기 위한 다양한 loss 함수가 있다. 몇 가지 예는 다음과 같다:

-  Cross-entropy loss:
   :math:`\mathcal{L} = - \log \sigma (y_{u,v}) - \sum_{v_i \sim P_n(v), i=1,\dots,k}\log \left[ 1 - \sigma (y_{u,v_i})\right]`
-  BPR loss:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} - \log \sigma (y_{u,v} - y_{u,v_i})`
-  Margin loss:
   :math:`\mathcal{L} = \sum_{v_i \sim P_n(v), i=1,\dots,k} \max(0, M - y_{u, v} + y_{u, v_i})`, 여기서 :math:`M` 은 상수 하이퍼-파라메터이다.

`implicit feedback <https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf>`__ 이나 `noise-contrastive estimation <http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf>`__ 를 알고 있다면, 이 아이디어는 친숙할 것이다.

:math:`u` 와 :math:`v` 사이의 점수를 계산하는 뉴럴 네트워크 모델은 :ref:`위에서 설명한 <guide_ko-training-edge-classification>`  에지 리그레션 모델과 동일하다.

다음은 dot product를 사용해서 에지들의 점수를 계산하는 예제이다.

.. code:: python

    class DotProductPredictor(nn.Module):
        def forward(self, graph, h):
            # h contains the node representations computed from the GNN defined
            # in the node classification section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
                return graph.edata['score']

학습 룹
~~~~~

점수를 예측하는 모델은 그래프들에 적용되기 때문에, 네가티브 샘들은 별도의 그래프로 표현되어야 한다. 즉, 그것은 에지들이 모두 네가티브 노드들의 쌍들로만 구성된 그래프이다.

아래 코드는 네가티브 샘들로 구성된 그래프를 만드는 예제이다. 각 에지 :math:`(u,v)` 는 :math:`k` 개의 네가티브 셈플들 :math:`(u,v_i)` 을 갖는다. 여기서 :math:`v_i` 는 균등 분포에서 샘플링된다.

.. code:: python

    def construct_negative_graph(graph, k):
        src, dst = graph.edges()
    
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
        return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())

에지 점수를 예측하는 모델은 에지 분류 또는 에지 리그레션 모델과 같다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.sage = SAGE(in_features, hidden_features, out_features)
            self.pred = DotProductPredictor()
        def forward(self, g, neg_g, x):
            h = self.sage(g, x)
            return self.pred(g, h), self.pred(neg_g, h)

그런 다음, 학습 룹은 반복적으로 네가티브 그래프를 만들고 loss를 계산한다.

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()
    
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

학습이 종료되면, 노드 representation은 다음과 같이 얻을 수 있다:

.. code:: python

    node_embeddings = model.sage(graph, node_features)

노드 임베딩을 사용하는 방법은 여러가지가 있다. 몇가지 예를 들면, 다운스트림 분류기 학습, 관련된 엔터리 추천을 위한 nearest neighbor search 또는 maximum inner product search와 같은 것이 있다.

Heterogeneous 그래프들
~~~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프에서의 링크 예측은 homogeneous 그래프에서의 링크 예측과 많이 다르지 않다. 다음 예제는 하나의 에지 타입에 대해서 예측을 수행한다고 가정하고 있는데, 이를 여러 에지 타입으로 확장하는 것은 쉽다.

링크 예측을 위해서 :ref:`앞에서 <guide_ko-training-edge-classification-heterogeneous-graph>` 의 ``HeteroDotProductPredictor`` 를 재활용해서 한 에지 타입에 대한 에지의 점수를 계산할 수 있다.

.. code:: python

    class HeteroDotProductPredictor(nn.Module):
        def forward(self, graph, h, etype):
            # h contains the node representations for each node type computed from
            # the GNN defined in the previous section (Section 5.1).
            with graph.local_scope():
                graph.ndata['h'] = h
                graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
                return graph.edges[etype].data['score']

네가티브 샘플링을 수행하기 위해서, 링크 예측을 수행할 에지 타입에 대한 네가티브 그램프를 생성하면 된다.

.. code:: python

    def construct_negative_graph(graph, k, etype):
        utype, _, vtype = etype
        src, dst = graph.edges(etype=etype)
        neg_src = src.repeat_interleave(k)
        neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,))
        return dgl.heterograph(
            {etype: (neg_src, neg_dst)},
            num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})

모델을 heterogeneous 그래프들에서 에지 분류하는 모델과는 약간 다른데, 그 이유는 링크 예측을 할 때 에지 타입을 지정해야하기 때문이다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, rel_names):
            super().__init__()
            self.sage = RGCN(in_features, hidden_features, out_features, rel_names)
            self.pred = HeteroDotProductPredictor()
        def forward(self, g, neg_g, x, etype):
            h = self.sage(g, x)
            return self.pred(g, h, etype), self.pred(neg_g, h, etype)

학습 룹은 homogeneous 그래프에 대한 학습 룹과 비슷하다.

.. code:: python

    def compute_loss(pos_score, neg_score):
        # Margin loss
        n_edges = pos_score.shape[0]
        return (1 - pos_score + neg_score.view(n_edges, -1)).clamp(min=0).mean()
    
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



