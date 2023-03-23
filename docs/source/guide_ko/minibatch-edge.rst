.. _guide_ko-minibatch-edge-classification-sampler:

6.2 이웃 샘플링을 사용한 에지 분류 GNN 모델 학습하기
-----------------------------------------

:ref:`(English Version) <guide-minibatch-edge-classification-sampler>`

에지 분류/리그레션 모델을 학습하는 것은 몇 가지 눈에 띄는 차이점이 있지만 노드 분류/리그레션과 어느정도 비슷하다.

이웃 샘플러 및 데이터 로더 정의하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`노드 분류에서 사용한 것과 같은 이웃 샘플러<guide_ko-minibatch-node-classification-sampler>` 를 사용할 수 있다.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)

에지 분류에 DGL이 제공하는 이웃 샘플러를 사용하려면, 미니-배치의 에지들의 집합을 iterate 하는 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 와 함께 사용해야한다. 이것은 아래 모듈에서 사용될 에지 미니-배치로부터 만들어질 서브 그래프와 *message flow graph* (MFG)들을 리턴한다.

다음 코드 예제는 PyTorch DataLoader를 만든다. 이는 베치들에 있는 학습 에지 ID 배열 :math:`train_eids` 들을 iterate 하고, 생성된 MFG들의 리스트를 GPU로 옮겨놓는다.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

.. note::

   Message flow graph의 개념은 :doc:`Stochastic Training Tutorial <tutorials/large/L0_neighbor_sampling_overview>` 를 참고하자.

   빌트인으로 지원되는 샘플러들에 대한 전체 목록은 :ref:`neighborhood sampler API reference <api-dataloading-neighbor-sampling>` 에 있다.

   :ref:`guide_ko-minibatch-customizing-neighborhood-sampler` 에는 여러분만의 이웃 샘플러 만드는 방법과 MFG 개념에 대한 보다 상세한 설명을 담고 있다.

이웃 샘플링을 위해서 원본 그래프에서 미니 배치의 에지들 제거하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

에지 분류 모델을 학습할 때, 때로는 computation dependency에서 학습 데이터에 있는 에지들을 존재하지 않았던 것처럼 만들기 위해 제거하는 것이 필요하다. 그렇지 않으면, 모델은 두 노드들 사이에 에지가 존재한다는 사실을 *인지* 할 것이고, 이 정보를 학습에 잠재적으로 이용할 수 있기 때문이다.

따라서, 에지 분류의 경우 때로는 이웃 샘플링은 미니-배치안에 샘플된 에지들 및 undirected 그래프인 경우 샘플된 에지의 역방향 에지들도 원본 그래프에서 삭제하기도 한다. :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 객체를 만들 때, ``exclude='reverse_id'`` 를 에지 ID와 그와 연관된 reverse 에지 ID들의 매핑 정보와 함께 지정할 수 있다.

.. code:: python

    n_edges = g.num_edges()
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
    
        # The following two arguments are specifically for excluding the minibatch
        # edges and their reverse edges from the original graph for neighborhood
        # sampling.
        exclude='reverse_id',
        reverse_eids=torch.cat([
            torch.arange(n_edges // 2, n_edges), torch.arange(0, n_edges // 2)]),
    
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

모델을 미니-배치 학습에 맞게 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

에지 분류 모델은 보통은 다음과 같이 두 부분으로 구성된다:

- 첫번째는 부속 노드(incident node)들의 representation을 얻는 부분
- 두번째는 부속 노드의 representation들로부터 에지 점수를 계산하는 부분

첫번째 부분은 :ref:`노드 분류<guide_ko-minibatch-node-classification-model>` 와 완전히 동일하기에, 단순하게 이를 재사용할 수 있다. 입력 DGL에서 제공하는 데이터 로더가 만들어 낸 MFG들의 리스트와 입력 피쳐들이 된다.

.. code:: python

    class StochasticTwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dglnn.GraphConv(in_features, hidden_features)
            self.conv2 = dglnn.GraphConv(hidden_features, out_features)
    
        def forward(self, blocks, x):
            x = F.relu(self.conv1(blocks[0], x))
            x = F.relu(self.conv2(blocks[1], x))
            return x

두번째 부분에 대한 입력은 보통은 이전 부분의 출력과 미니배치의 에지들에 의해서 유도된 원본 그래프의 서브 그래프가 된다. 서브 그래프는 같은 데이터 로더에서 리턴된다. :meth:`dgl.DGLGraph.apply_edges` 를 사용해서 에지 서브 그래프를 사용해서 에지들의 점수를 계산한다.

다음 코드는 부속 노드 피처들을 연결하고, 이를 dense 레이어에 입력해서 얻은 결과로 에지들의 점수를 예측하는 예를 보여준다.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def apply_edges(self, edges):
            data = torch.cat([edges.src['x'], edges.dst['x']], 1)
            return {'score': self.W(data)}
    
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                edge_subgraph.apply_edges(self.apply_edges)
                return edge_subgraph.edata['score']

전체 모델은 아래와 같이 데이터 로더로부터 얻은 MFG들의 리스트와 에지 서브 그래프, 그리고 입력 노드 피쳐들을 사용한다.

.. code:: python

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes):
            super().__init__()
            self.gcn = StochasticTwoLayerGCN(
                in_features, hidden_features, out_features)
            self.predictor = ScorePredictor(num_classes, out_features)
    
        def forward(self, edge_subgraph, blocks, x):
            x = self.gcn(blocks, x)
            return self.predictor(edge_subgraph, x)

DGL에서는 에지 서브 그래프의 노드들이 MFG들의 리스트에서 마지막 MFG의 출력 노드들과 동일하도록 확인한다.

학습 룹
~~~~~

학습 룹은 노드 분류의 학습 룹과 비슷하다. 데이터 로더를 iterate해서, 미니배치의 에지들에 의해서 유도된 서브 그래프와 에지들의 부속 노드(incident node)들의 representation들을 계산하기 위한 MFG들의 목록을 얻는다.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        edge_labels = edge_subgraph.edata['labels']
        edge_predictions = model(edge_subgraph, blocks, input_features)
        loss = compute_loss(edge_labels, edge_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

Heterogeneous 그래프의 경우
~~~~~~~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프들의 노드 representation들을 계산하는 모델은 에지 분류/리그레션을 위한 부속 노드 representation들을 구하는데 사용될 수 있다.

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

점수를 예측하기 위한 homogeneous 그래프와 heterogeneous 그래프간의 유일한 구현상의 차이점은 :meth:`~dgl.DGLGraph.apply_edges` 를 호출할 때 에지 타입들을 사용한다는 점이다.

.. code:: python

    class ScorePredictor(nn.Module):
        def __init__(self, num_classes, in_features):
            super().__init__()
            self.W = nn.Linear(2 * in_features, num_classes)
    
        def apply_edges(self, edges):
            data = torch.cat([edges.src['x'], edges.dst['x']], 1)
            return {'score': self.W(data)}
    
        def forward(self, edge_subgraph, x):
            with edge_subgraph.local_scope():
                edge_subgraph.ndata['x'] = x
                for etype in edge_subgraph.canonical_etypes:
                    edge_subgraph.apply_edges(self.apply_edges, etype=etype)
                return edge_subgraph.edata['score']

    class Model(nn.Module):
        def __init__(self, in_features, hidden_features, out_features, num_classes,
                     etypes):
            super().__init__()
            self.rgcn = StochasticTwoLayerRGCN(
                in_features, hidden_features, out_features, etypes)
            self.pred = ScorePredictor(num_classes, out_features)

        def forward(self, edge_subgraph, blocks, x):
            x = self.rgcn(blocks, x)
            return self.pred(edge_subgraph, x)

데이터 로더 구현도 노드 분류을 위한 것과 아주 비슷하다. 유일한 차이점은 :class:`~dgl.dataloading.pytorch.NodeDataLoader` 대신에 :class:`~dgl.dataloading.pytorch.EdgeDataLoader` 를 사용하고, 노드 타입과 노드 ID 텐서들의 사전 대신에 에지 타입과 에지 ID 텐서들의 사전을 사용한다는 것이다.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

만약 heterogeneous 그래프에서 역방향의 에지를 배제하고자 한다면 약간 달라진다. Heterogeneous 그래프에서 역방향 에지들은 에지와는 다른 에지 타입을 갖는 것이 보통이다. 이는 “forward”와 “backward” 관계들을 구분직기 위해서이다. (즉, ``follow`` 와 ``followed by`` 는 서로 역 관계이고, ``purchase`` 와 ``purchased by`` 는 서로 역 관계인 것 처럼)

만약 어떤 타입의 에지들이 다른 타입의 같은 ID를 갖는 역방향 에지를 갖는다면, 에지 타입들과 
그것들의 반대 타입간의 매핑을 명시할 수 있다. 미니배치에서 에지들과 그것들의 역방향 에지를 배제하는 것은
다음과 같다.

.. code:: python

    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_eid_dict, sampler,
    
        # The following two arguments are specifically for excluding the minibatch
        # edges and their reverse edges from the original graph for neighborhood
        # sampling.
        exclude='reverse_types',
        reverse_etypes={'follow': 'followed by', 'followed by': 'follow',
                        'purchase': 'purchased by', 'purchased by': 'purchase'}
    
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

학습 룹은 ``compute_loss`` 의 구현이 노드 타입들과 예측 값에 대한 두 사전들을 인자로 받는다는 점을 제외하면,
homogeneous 그래프의 학습 룹 구현과 거의 같다.

.. code:: python

    model = Model(in_features, hidden_features, out_features, num_classes, etypes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, edge_subgraph, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        edge_subgraph = edge_subgraph.to(torch.device('cuda'))
        input_features = blocks[0].srcdata['features']
        edge_labels = edge_subgraph.edata['labels']
        edge_predictions = model(edge_subgraph, blocks, input_features)
        loss = compute_loss(edge_labels, edge_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

`GCMC <https://github.com/dmlc/dgl/tree/master/examples/pytorch/gcmc>`__ 은 이분 그래프(bipartite graph)에 대한 에지 분류 예제이다.

