.. _guide_ko-minibatch-link-classification-sampler:

6.3 이웃 샘플링을 사용한 링크 예측 GNN 모델 학습하기
-----------------------------------------

:ref:`(English Version) <guide-minibatch-link-classification-sampler>`

Negative 샘플링을 사용한 이웃 샘플러 및 데이터 로더 정의하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

노드/에지 분류에서 사용한 이웃 샘플러를 그대로 사용하는 것이 가능하다.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

DGL의 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 는 링크 예측를 위한 negative 샘플 생성을
지원한다. 이를 사용하기 위해서는, negative 샘플링 함수를 제공해야한다. :class:`~dgl.dataloading.negative_sampler.Uniform` 은 uniform 샘플링을 해주는 함수이다. 에지의 각 소스 노드에 대해서,이 함수는 ``k`` 개의 negative 목적지 노드들을 샘플링한다.

아래 코드는 에지의 각 소스 노드에 대해서 5개의 negative 목적지 노드를 균등하게 선택한다.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

빌드인 negative 샘플러들은 :ref:`api-dataloading-negative-sampling` 에서 확인하자.

직접 만든 negative 샘플러 함수를 사용할 수도 있다. 이 함수는 원본 그래프 ``g`` 와, 미니배치 에지 ID 배열 ``eid`` 를 받아서
소스 ID 배열과 목적지 ID 배열의 쌍을 리턴해야 한다.

아래 코드 예제는 degree의 거듭제곱에 비례하는 확률 분포에 따라서 negative 목적지 노드들을 샘플링하는 custom negative 샘플러다.

.. code:: python

    class NegativeSampler(object):
        def __init__(self, g, k):
            # caches the probability distribution
            self.weights = g.in_degrees().float() ** 0.75
            self.k = k
    
        def __call__(self, g, eids):
            src, _ = g.find_edges(eids)
            src = src.repeat_interleave(self.k)
            dst = self.weights.multinomial(len(src), replacement=True)
            return src, dst
    
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler,
        negative_sampler=NegativeSampler(g, 5),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

모델을 미니-배치 학습에 맞게 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`guide_ko-training-link-prediction` 에서 설명한 것처럼, 링크 예측은 (positive 예제인) 에지의 점수와 존재하지 않는 에지(즉, negative 예제)의 점수를 비교하는 것을 통해서 학습될 수 있다. 에지들의 점수를 계산하기 위해서, 에지 분류/리그레션에서 사용했던 노드 representation 계산 모델을 재사용한다.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
            self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
    
        def forward(self, blocks, x):
            x = F.relu(self.conv1(blocks[0], x))
            x = F.relu(self.conv2(blocks[1], x))
            return x

점수 예측을 위해서 확률 분포 대신 각 에지의 scalar 점수를 예측하기만 하면되기 때문에, 이 예제는 부속 노드 representation들의 dot product로 점수를 계산하는 방법을 사용한다.

.. code:: python

    class ScorePredictor(nn.Module):
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                edge_subgraph.apply_edges(dgl.function.u_dot_v('x', 'x', 'score'))
                return edge_subgraph.edata['score']

Negative 샘플러가 지정되면, DGL의 데이터 로더는 미니배치 마다 다음 3가지 아이템들을 만들어낸다.

- 샘플된 미니배치에 있는 모든 에지를 포함한 postive 그래프
- Negative 샘플러가 생성한 존재하지 않는 에지 모두를 포함한 negative 그래프
- 이웃 샘플러가 생성한 *message flow graph* (MFG)들의 리스트

이제 3가지 아이템와 입력 피쳐들을 받는 링크 예측 모델을 다음과 같이 정의할 수 있다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.gcn = StochasticTwoLayerGCN(
                in_features, hidden_features, out_features)
    
        def forward(self, positive_graph, negative_graph, blocks, x):
            x = self.gcn(blocks, x)
            pos_score = self.predictor(positive_graph, x)
            neg_score = self.predictor(negative_graph, x)
            return pos_score, neg_score

학습 룹
~~~~~

학습 룹은 데이터 로더를 iterate하고, 그래프들과 입력 피쳐들을 위해서 정의한 모델에 입력하는 것일 뿐이다.

.. code:: python

    def compute_loss(pos_score, neg_score):
        # an example hinge loss
        n = pos_score.shape[0]
        return (neg_score.view(n, -1) - pos_score.view(n, -1) + 1).clamp(min=0).mean()

    model = Model(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()

DGL에서는 homogeneous 그래프들에 대한 링크 예측의 예제로 `unsupervised learning GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_sampling_unsupervised.py>`__ 를 제공한다.

Heterogeneous 그래프의 경우
~~~~~~~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프들의 노드 representation들을 계산하는 모델은 에지 분류/리그레션을 위한 부속 노드
representation들을 구하는데 사용될 수 있다.

.. code:: python

    class StochasticTwoLayerRGCN(nn.Module):
        def __init__(self, in_feat, hidden_feat, out_feat, rel_names):
            super().__init__()
            self.conv1 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(in_feat, hidden_feat, norm='right')
                    for rel in rel_names
                })
            self.conv2 = dglnn.HeteroGraphConv({
                    rel : dglnn.GraphConv(hidden_feat, out_feat, norm='right')
                    for rel in rel_names
                })
    
        def forward(self, blocks, x):
            x = self.conv1(blocks[0], x)
            x = self.conv2(blocks[1], x)
            return x

점수를 예측하기 위한 homogeneous 그래프와 heterogeneous 그래프간의 유일한 구현상의 차이점은
:meth:`dgl.DGLGraph.apply_edges` 를 호출할 때 에지 타입들을 사용한다는 점이다.

.. code:: python

    class ScorePredictor(nn.Module):
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(
                        dgl.function.u_dot_v('x', 'x', 'score'), etype=etype)
                return edge_subgraph.edata['score']

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes,
                     etypes):
            super().__init__()
            self.rgcn = StochasticTwoLayerRGCN(
                in_features, hidden_features, out_features, etypes)
            self.pred = ScorePredictor()

        def forward(self, positive_graph, negative_graph, blocks, x):
            x = self.rgcn(blocks, x)
            pos_score = self.pred(positive_graph, x)
            neg_score = self.pred(negative_graph, x)
            return pos_score, neg_score

데이터 로더 구현도 노드 분류을 위한 것과 아주 비슷하다. 유일한 차이점은 negative 샘플러를 사용하며, 노드 타입과 노드 ID 텐서들의 사전 대신에 에지 타입과 에지 ID 텐서들의 사전을 사용한다는 것이다.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        negative_sampler=dgl.dataloading.negative_sampler.Uniform(5),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

만약 직접 만든 negative 샘플링 함수를 사용하기를 원한다면, 그 함수는 원본 그래프, 에지 타입과 에지 ID 텐서들의 dictionary를 인자로 받아야하고, 에지 타입들과 소스-목적지 배열 쌍의 dictionary를 리턴해야한다. 다음은 예제 함수이다.

.. code:: python

   class NegativeSampler(object):
       def __init__(self, g, k):
           # caches the probability distribution
           self.weights = {
               etype: g.in_degrees(etype=etype).float() ** 0.75
               for etype in g.canonical_etypes}
           self.k = k

       def __call__(self, g, eids_dict):
           result_dict = {}
           for etype, eids in eids_dict.items():
               src, _ = g.find_edges(eids, etype=etype)
               src = src.repeat_interleave(self.k)
               dst = self.weights[etype].multinomial(len(src), replacement=True)
               result_dict[etype] = (src, dst)
           return result_dict

다음으로는 에지 타입들와 에지 ID들의 dictionary와 negative 샘플러를 데이터 로더에 전달한다. 예를 들면, 아래 코드는 heterogeneous 그래프의 모든 에지들을 iterate하는 예이다.

.. code:: python

    train_eid_dict = {
        etype: g.edges(etype=etype, form='eid')
        for etype in g.canonical_etypes}

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        negative_sampler=NegativeSampler(g, 5),
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

학습 룹은 ``compute_loss`` 의 구현이 노드 타입들과 예측 값에 대한 두 사전들을 인자로 받는다는 점을 제외하면, homogeneous 그래프의 학습 룹 구현과 거의 같다.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes, etypes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, positive_graph, negative_graph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        positive_graph = positive_graph.to(torch.device('cuda'))
        negative_graph = negative_graph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        pos_score, neg_score = model(positive_graph, negative_graph, blocks, input_features)
        loss = compute_loss(pos_score, neg_score)
        opt.zero_grad()
        loss.backward()
        opt.step()



