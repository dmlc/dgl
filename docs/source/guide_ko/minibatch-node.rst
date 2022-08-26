.. _guide_ko-minibatch-node-classification-sampler:

6.1 이웃 샘플링을 사용한 노드 분류 GNN 모델 학습하기
-----------------------------------------

:ref:`(English Version) <guide-minibatch-node-classification-sampler>`

Stochastic 학습이 되도록 모델을 만들기 위해서는, 다음과 같은 것이 필요하다.

- 이웃 샘플러 정의하기
- 미니 배치 학습이 되도록 모델을 변경하기
- 학습 룹 고치기

이제, 이 단계를 어떻게 구현하는 하나씩 살펴보자.

이웃 샘플러 및 데이터 로더 정의하기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

DGL는 계산하기를 원하는 노드들에 대해서 각 레이어에서 필요한 computation dependency들을 생성하는 몇 가지 이웃 샘플러 클래스들을 가지고 있다.

가장 단순한 이웃 샘플러는 :class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` 로, 노드가 그 노드의 모든 이웃들로부터 메시지를 수집하도록 해준다.

DGL의 샘플러를 사용하기 위해서는 이를 미니배치에 있는 노드들의 집한은 iterate하는 :class:`~dgl.dataloading.pytorch.NodeDataLoader` 와 합쳐야한다.

다음 예제 코드는 배치들의 학습 노드 ID 배열 ``train_nids`` 를 iterate하고, 생성된 MFG(Message Flow Graph)들의 목록을 GPU로 옮기는 PyTorch DataLoader를 만든다.

.. code:: python

    import dgl
    import dgl.nn as dglnn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

DataLoader를 iterate 하면서 각 레이어에 대한 computation dependency들을 대표하도록 특별하게 생성된 그래프들의 리스트를 얻을 수 있다. DGL에서 이것들을 *message flow graph* (MFG) 라고 부른다.

.. code:: python

    input_nodes, output_nodes, blocks = next(iter(dataloader))
    print(blocks)

Iterator는 매번 세개의 아이템을 생성한다. ``input_nodes`` 는 ``output_nodes`` 의 representation을 계산하는데 필요한 노드들을 담고 있다. ``block`` 은 그것의 노드가 출력으로 계산되어야 할 각 GNN 레이어에 대해 어떤 노드 representation들이 입력으로 필요한지, 입력 노드들의 representation들이 출력 노드로 어떻게 전파되어야 하는지를 설명한다.

.. note::

   Message flow graph의 개념은 :doc:`Stochastic Training Tutorial <tutorials/large/L0_neighbor_sampling_overview>` 을 참고하자.

   지원되는 빌드인 샘플러들의 전체 목록은 :ref:`neighborhood sampler API reference <api-dataloading-neighbor-sampling>` 에서 찾아볼 수 있다.

   :ref:`guide_ko-minibatch-customizing-neighborhood-sampler` 에는 여러분만의 이웃 샘플러 만드는 방법과 MFG 개념에 대한 보다 상세한 설명을 담고 있다.


.. _guide_ko-minibatch-node-classification-model:

모델을 미니-배치 학습에 맞게 만들기
~~~~~~~~~~~~~~~~~~~~~~~~~~~

만약 DGL에서 제공하는 메시지 전달 모듈만을 사용하고 있다면, 모델을 미니-배치 학습에 맞도록 수정할 것은 적다. 멀티-레이어 GCN을 예로 들어보자. 그래프 전체에 대한 모델 구현은 아래와 같다.

.. code:: python

    class TwoLayerGCN(nn.Module):
        def __init__(self, in_features, hidden_features, out_features):
            super().__init__()
            self.conv1 = dglnn.GraphConv(in_features, hidden_features)
            self.conv2 = dglnn.GraphConv(hidden_features, out_features)
    
        def forward(self, g, x):
            x = F.relu(self.conv1(g, x))
            x = F.relu(self.conv2(g, x))
            return x

이 때, 변경해야할 것은 ``g`` 를 앞에서 생성된 ``block`` 로 교체하는 것이 전부이다.

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

위 DGL ``GraphConv`` 모듈들은 데이터 로더가 생성한 ``block`` 의 원소를 argument로 받는다.

:ref:`The API reference of each NN module <apinn>` 는 모듈이 MFG를 argument로 받을 수 있는지 없는지를 알려주고 있다.

만약 여러분 자신의 메시지 전달 모듈을 사용하고 싶다면, :ref:`guide_ko-minibatch-custom-gnn-module` 를 참고하자.

학습 룹
~~~~~

단순하게 학습 룹은 커스터마이징된 배치 iterator를 사용해서 데이터셋을 iterating하는 것으로 구성된다. MFG들의 리스트를 반환하는 매 iteration마다, 다음과 같은 일을 한다.

1. 입력 노드들의 노드 피처들을 GPU로 로딩한다. 노드 피쳐들은 메모리나 외부 저장소에 저장되어 있을 수 있다. 그래프 전체 학습에서 모든 노드들의 피처를 로드하는 것과는 다르게, 입력 노드들의 피처만 로드하면 된다는 점을 유의하자.
   

   만약 피쳐들이 ``g.ndata`` 에 저장되어 있다면, 그 피쳐들은 ``blocks[0].srcdata`` 에 저장된 피쳐들, 즉 첫번째 MFG의 소스 노드들의 피처들을 접근해서 로드될 수 있다. 여기서 노드들은 최종 representation을 계산하는데 필요한 모든 노드들을 의미한다.

2. MFG들의 리스트 및 입력 노드 피쳐들을 멀티-레이어 GNN에 입력해서 결과를 
얻는다.

3. 출력 노드에 해당하는 노드 레이블을 GPU에 로드한다. 비슷하게, 노드 레이블은 메모리나 외부 저장소에 저장되어 있을 수 있다. 역시, 그래프 전체 학습에서 모든 노드들의 레이블을 로드하는 것과는 다르게, 출력 노드들의 레이블만 로드한다는 점을 알아두자.
   
   피처가 ``g.ndata`` 에 저장되어 있다면, 레이블은 ``blocks[-1].dstdata`` 의 피쳐들 즉, 마지막 MFG의 목적지 노드들의 피쳐들을 접근해서 로드될 수 있다. 이것들은 최종 representation을 계산할 노드들과 같다.

4. loss를 계산한 후, backpropagate를 수행한다.

.. code:: python

    model = StochasticTwoLayerGCN(in_features, hidden_features, out_features)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata['features']
        output_labels = blocks[-1].dstdata['label']
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

DGL에서는 end-to-end stochastic 학습 예제인 `GraphSAGE
implementation <https://github.com/dmlc/dgl/blob/master/examples/pytorch/graphsage/node_classification.py>`__ 를 제공한다.

Heterogeneous 그래프의 경우
~~~~~~~~~~~~~~~~~~~~~~~~

Heterogeneous 그래프에 대한 노드 분류 그래프 뉴럴 네트워크를 학습하는 것은 간단하다.

:ref:`how to train a 2-layer RGCN on full graph <guide_ko-training-rgcn-node-classification>` 를 예로 들어보자. 미니-배치 학습을 하는 RGCN 구현 코드는 이 예제와 매우 비슷하다. (간단하게 하기 위해서 self-loop, non-linearity와 기본적인 decomposition은 제거했다.)

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

또한, DGL이 제공하는 일부 샘플러들은 heterogeneous 그래프를 지원한다. 예를 들어, 제공되는 :class:`~dgl.dataloading.neighbor.MultiLayerFullNeighborSampler` 클래스 및 :class:`~dgl.dataloading.pytorch.NodeDataLoader` 클래스를 stochastic 학습에도 여전히 사용할 수 있다. 전체 이웃 샘플링에서 다른 점은 학습 셋에 노드 타입들과 노드 ID들의 사전을 명시해야한다는 것 뿐이다.

.. code:: python

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nid_dict, sampler,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=4)

학습 룹은 homogeneous 그래프에 대한 학습 룹이랑 거의 유사하다. 다른 점은 ``compute_loss`` 의 구현에서 노드 타입들와 예측 결과라는 두개의 dictionary들을 인자로 받는다는 것이다.

.. code:: python

    model = StochasticTwoLayerRGCN(in_features, hidden_features, out_features, etypes)
    model = model.cuda()
    opt = torch.optim.Adam(model.parameters())
    
    for input_nodes, output_nodes, blocks in dataloader:
        blocks = [b.to(torch.device('cuda')) for b in blocks]
        input_features = blocks[0].srcdata     # returns a dict
        output_labels = blocks[-1].dstdata     # returns a dict
        output_predictions = model(blocks, input_features)
        loss = compute_loss(output_labels, output_predictions)
        opt.zero_grad()
        loss.backward()
        opt.step()

End-to-end stochastic 학습 예제는 `RGCN
implementation <https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn-hetero/entity_classify_mb.py>`__ 를 참고하자.


