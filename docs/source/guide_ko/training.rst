.. _guide_ko-training:

5장: 그래프 뉴럴 네트워크 학습하기
==========================

:ref:`(English Version) <guide-training>`

개요
----------------

이 장에서는 :ref:`guide_ko-message-passing` 에서 소개한 메시지 전달 방법과 :ref:`guide_ko-nn` 에서 소개한 뉴럴 네트워크 모듈을 사용해서 작은 그래프들에 대한 노드 분류, 에지 분류, 링크 예측, 그리고 그래프 분류를 위한 그래프 뉴럴 네트워크를 학습하는 방법에 대해서 알아본다.

여기서는 그래프 및 노드 및 에지 피쳐들이 GPU 메모리에 들어갈 수 있는 크기라고 가정한다. 만약 그렇지 않다면, :ref:`guide_ko-minibatch` 를 참고하자.

그리고, 그래프와 노드/에지 피쳐들은 이미 프로세싱되어 있다고 가정한다. 만약 DGL에서 제공되는 데이터셋 또는 :ref:`guide_ko-data-pipeline` 에서 소개한 ``DGLDataset`` 과 호환되는 다른 데이터셋을 사용할 계획이라면, 다음과 같이 단일-그래프 데이터셋을 위한 그래프를 얻을 수 있다.

.. code:: python

    import dgl
    
    dataset = dgl.data.CiteseerGraphDataset()
    graph = dataset[0]

주의: 이 장의 예제들은 PyTorch를 백엔드로 사용한다.

.. _guide_ko-training-heterogeneous-graph-example:

Heterogeneous 그래프
~~~~~~~~~~~~~~~~~~

때로는 heterogeneous 그래프를 사용할 경우도 있다. 노드 분류, 에지 분류, 그리고 링크 예측 과제들의 예제를 위해서 임의로 만든 heterogeneous 그래프를 사용하겠다.

임의로 생성한 heterogeneous 그래프 ``hetero_graph`` 는 다음과 같은 에지 타입을 갖는다:

-  ``('user', 'follow', 'user')``
-  ``('user', 'followed-by', 'user')``
-  ``('user', 'click', 'item')``
-  ``('item', 'clicked-by', 'user')``
-  ``('user', 'dislike', 'item')``
-  ``('item', 'disliked-by', 'user')``

.. code:: python

    import numpy as np
    import torch
    
    n_users = 1000
    n_items = 500
    n_follows = 3000
    n_clicks = 5000
    n_dislikes = 500
    n_hetero_features = 10
    n_user_classes = 5
    n_max_clicks = 10
    
    follow_src = np.random.randint(0, n_users, n_follows)
    follow_dst = np.random.randint(0, n_users, n_follows)
    click_src = np.random.randint(0, n_users, n_clicks)
    click_dst = np.random.randint(0, n_items, n_clicks)
    dislike_src = np.random.randint(0, n_users, n_dislikes)
    dislike_dst = np.random.randint(0, n_items, n_dislikes)
    
    hetero_graph = dgl.heterograph({
        ('user', 'follow', 'user'): (follow_src, follow_dst),
        ('user', 'followed-by', 'user'): (follow_dst, follow_src),
        ('user', 'click', 'item'): (click_src, click_dst),
        ('item', 'clicked-by', 'user'): (click_dst, click_src),
        ('user', 'dislike', 'item'): (dislike_src, dislike_dst),
        ('item', 'disliked-by', 'user'): (dislike_dst, dislike_src)})
    
    hetero_graph.nodes['user'].data['feature'] = torch.randn(n_users, n_hetero_features)
    hetero_graph.nodes['item'].data['feature'] = torch.randn(n_items, n_hetero_features)
    hetero_graph.nodes['user'].data['label'] = torch.randint(0, n_user_classes, (n_users,))
    hetero_graph.edges['click'].data['label'] = torch.randint(1, n_max_clicks, (n_clicks,)).float()
    # randomly generate training masks on user nodes and click edges
    hetero_graph.nodes['user'].data['train_mask'] = torch.zeros(n_users, dtype=torch.bool).bernoulli(0.6)
    hetero_graph.edges['click'].data['train_mask'] = torch.zeros(n_clicks, dtype=torch.bool).bernoulli(0.6)


로드맵
----

이 장은 그래프 학습 테스크를 설명하기 위해서 4개의 절로 구성되어 있다.

* :ref:`guide_ko-training-node-classification`
* :ref:`guide_ko-training-edge-classification`
* :ref:`guide_ko-training-link-prediction`
* :ref:`guide_ko-training-graph-classification`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    training-node
    training-edge
    training-link
    training-graph
