.. _guide_ko-training-node-classification:

5.1 노드 분류/리그래션(Regression)
--------------------------------------------------

:ref:`(English Version) <guide-training-node-classification>`

가장 유명하고 널리 적용되고 있는 그래프 뉴럴 네트워크 중에 하나가 노드 분류이다. 학습/검증/테스트 셋의 각 노드는 미리 정해진 카테로기들로 중에 하나를 ground truth 카테고리로 분류되어 있다. 노드 regression도 비슷하다. 학습/검증/테스트 셋의 각 노드에 ground truth 수가 할당되어 있다.

개요
~~~~~~

노드를 분류하기 위해서 그래프 뉴럴 네트워크는 :ref:`guide_ko-message-passing` 에서 소개한 메시지 전달 방법을 수행해서 노드 자신의 피쳐 뿐만 아니라 그 노드의 이웃 노드 및 에지의 피쳐도 함께 활용한다. 메시지 전달은 여러 회 반복해서 더 큰 범위의 이웃들에 대한 정보를 활용할 수 있다.

뉴럴 네트워크 모델 작성하기
~~~~~~~~~~~~~~~~~~~~

DGL은 한 차례 메시지 전달을 수행하는 몇 가지 빌트인 graph convolution 모듈을 제공한다. 여기서 우리는 GraphSAGE에서 사용되는 graph convolution 모듈인 :class:`dgl.nn.pytorch.SAGEConv` (MXNet과 TensorFlow에서도 사용 가능)를 사용한다.

보통 그래프에 대한 딥러닝 모델에서는 메시지 전달이 여러 번 수행되는 멀티-레이어 그래프 뉴럴 네트워크가 필요하다. 이는 다음 코드처럼 graph convolution 모듈들을 쌓아서 구현할 수 있다.

.. code:: python

    # Contruct a two-layer GNN model
    import dgl.nn as dglnn
    import torch.nn as nn
    import torch.nn.functional as F
    class SAGE(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats):
            super().__init__()
            self.conv1 = dglnn.SAGEConv(
                in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
            self.conv2 = dglnn.SAGEConv(
                in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = F.relu(h)
            h = self.conv2(graph, h)
            return h

위 모델은 노드 분류 뿐만 아니라, :ref:`guide_ko-training-edge-classification` , :ref:`guide_ko-training-link-prediction` , 또는 :ref:`guide_ko-training-graph-classification` 와 같은 다른 다운스트림 테스크들을 위한 히든 노드 표현을 구하기 위해서 사용될 수 있음을 알아두자.

빌트인 graph convolution 모듈의 전체 목록은 :ref:`apinn` 를 참고하자.

DGL 뉴럴 네트워크 모듈이 어떻게 동작하는지 그리고 메시지 전달을 활용한 커스텀 뉴럴 네트워크 모듈을 작성하는 방법은 :ref:`guide_ko-nn` 에 있는 예제들을 참고하자.

학습 룹(loop)
~~~~~~~~~~~

전체 그래프를 이용한 학습은 단지 위에서 정의된 모델에 forward propagation 그리고 학습 노드들의 groud truth 레이블과 예측을 비교해서 loss를 계산하는 것으로 구성된다.

이 절은 빌드인 데이터셋 :class:`dgl.data.CiteseerGraphDataset` 을 사용해서 학습 룹을 설명한다. 노드 피처 및 레이블은 각 그래프 인스턴스에 저장되어 있고, 학습-검증-테스트 분할 또한 그래프에 이진 마스크로서 저장되어 있다. 이는 :ref:`guide_ko-data-pipeline` 에서 본것과 비슷하다.

.. code:: python

    node_features = graph.ndata['feat']
    node_labels = graph.ndata['label']
    train_mask = graph.ndata['train_mask']
    valid_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']
    n_features = node_features.shape[1]
    n_labels = int(node_labels.max().item() + 1)

다음은 정확도(accuracy)로 모델을 평가하는 예제 코드이다.

.. code:: python

    def evaluate(model, graph, features, labels, mask):
        model.eval()
        with torch.no_grad():
            logits = model(graph, features)
            logits = logits[mask]
            labels = labels[mask]
            _, indices = torch.max(logits, dim=1)
            correct = torch.sum(indices == labels)
            return correct.item() * 1.0 / len(labels)

그리고, 학습 룹은 다음과 같이 작성할 수 있다.

.. code:: python

    model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(10):
        model.train()
        # forward propagation by using all nodes
        logits = model(graph, node_features)
        # compute loss
        loss = F.cross_entropy(logits[train_mask], node_labels[train_mask])
        # compute validation accuracy
        acc = evaluate(model, graph, node_features, node_labels, valid_mask)
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in this example.

`GraphSAGE <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/train_full.py>`__ 는 end-to-end homogeneous 그래프 노드 분류 예제를 제공한다. 해당 모델은 ``GraphSAGE`` 클래스에 구현되어 있고, 조정가능 한 레이어 수, dropout 확률들, 그리고 커스터마이징이 가능한 aggregation 함수 및 비선형성 등의 예제가 포함되어 있다.

.. _guide_ko-training-rgcn-node-classification:

Heterogeneous 그래프
~~~~~~~~~~~~~~~~~~

만약 그래프가 heterogeneous(이종)이라면, 여러분은 노드의 모든 에지 타입에 대한 이웃들로부터 메시지를 수집하기를 원할 것이다. 모든 에지 종류에 대해서 각 에지 타입별로 서로 다른 graph convolution 모듈을 사용한 메시지 전달을 수행하는 것은, :class:`dgl.nn.pytorch.HeteroGraphConv` (MXNet과 Tensorflow에서도 제공함) 모듈을 사용해서 가능하다.

아래 코드는 heterogeneous graph convolution을 정의하는데, 이는 각 에지 타입에 따라 별도의 graph convolution을 수행하고, 모든 노드 타입들에 대한 결과로서 각 에지 타입에 대한 메시지 aggregation 값들을 합하는 일을 수행한다.

.. code:: python

    # Define a Heterograph Conv model

    class RGCN(nn.Module):
        def __init__(self, in_feats, hid_feats, out_feats, rel_names):
            super().__init__()
            
            self.conv1 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(in_feats, hid_feats)
                for rel in rel_names}, aggregate='sum')
            self.conv2 = dglnn.HeteroGraphConv({
                rel: dglnn.GraphConv(hid_feats, out_feats)
                for rel in rel_names}, aggregate='sum')
      
        def forward(self, graph, inputs):
            # inputs are features of nodes
            h = self.conv1(graph, inputs)
            h = {k: F.relu(v) for k, v in h.items()}
            h = self.conv2(graph, h)
            return h

``dgl.nn.HeteroGraphConv`` 는 노드 타입들과 노드 피쳐 텐서들의 사전을 입력으로 받고, 노드 타입과 노드 피쳐의 다른 사전을 리턴한다.

여기서 사용되는 데이터셋은 이미 user 및 item 피쳐를 가지고 있고, 이는 :ref:`heterogeneous graph example <guide_ko-training-heterogeneous-graph-example>` 에서 확인할 수 있다.

.. code:: python

    model = RGCN(n_hetero_features, 20, n_user_classes, hetero_graph.etypes)
    user_feats = hetero_graph.nodes['user'].data['feature']
    item_feats = hetero_graph.nodes['item'].data['feature']
    labels = hetero_graph.nodes['user'].data['label']
    train_mask = hetero_graph.nodes['user'].data['train_mask']

Forward propagation을 다음과 같이 단순하게 실행된다.

.. code:: python

    node_features = {'user': user_feats, 'item': item_feats}
    h_dict = model(hetero_graph, {'user': user_feats, 'item': item_feats})
    h_user = h_dict['user']
    h_item = h_dict['item']

학습 룹은 예측을 계산할 노드 representation들의 사전을 사용하는 것을 제외하고는 homogeneous graph의 학습 룹과 동일하다. 예를 들어, ``user`` 노드 만을 예측하고 싶다면, 단지 리턴된 사전에서 ``user`` 노드 임베딩을 추출하면 된다.

.. code:: python

    opt = torch.optim.Adam(model.parameters())
    
    for epoch in range(5):
        model.train()
        # forward propagation by using all nodes and extracting the user embeddings
        logits = model(hetero_graph, node_features)['user']
        # compute loss
        loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        # Compute validation accuracy.  Omitted in this example.
        # backward propagation
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss.item())
    
        # Save model if necessary.  Omitted in the example.

DGL은 `RGCN <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify.py>`__ 의 end-to-end 예제를 제공한다. Heterogeneous graph convolution의 정의는 `모델 구현 파일 <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/model.py>`__ ``RelGraphConvLayer`` 에서 확인할 수 있다.

