.. _guide_ko-graph-graphs-nodes-edges:

1.2 그래프, 노드, 그리고 에지
----------------------------

:ref:`(English Version)<guide-graph-graphs-nodes-edges>`

DGL은 각 노드에 고유한 번호를 부여하는데 이를 노드 ID라고 하고, 각 에지에는 연결된 노드의 ID들에 해당하는 번호 쌍으로 표현된다. DGL은 각 에지에 고유한 번호를 부여하고, 이를 **에지 ID**라고 하며, 그래프에 추가된 순서에 따라 번호가 부여된다. 노드와 에지 ID의 번호는 0부터 시작한다. DGL에서는 모든 에지는 방향을 갖고, 에지 :math:`(u,v)` 는 노드 :math:`u` 에서 노드 :math:`v` 로 이어진 방향을 나타낸다.

여러 노드를 표현하기 위해서 DGL는 노드 ID로 1차원 정수 텐서를 사용한다. (PyTorch의 tensor, TensorFlow의 Tensor, 또는 MXNet의 ndarry) DGL은 이 포멧을 "노드-텐서"라고 부른다. DGL에서 에지들은 노드-텐서의 튜플 :math:`(U, V)` 로 표현된다. :math:`(U[i], V[i])`  는 :math:`U[i]` 에서 :math:`V[i]` 로의 에지이다. 

:class:`~dgl.DGLGraph` 를 만드는 방법 중의 하나는 :func:`dgl.graph` 메소드를 사용하는 것이다. 이는 에지 집합을 입력으로 받는다. 또한 DGL은 다른 데이터 소스로부터 그래프들을 생성하는 것도 지원한다. :ref:`guide_ko-graph-external` 참고하자.

다음 코드는 아래와 같은 4개의 노드를 갖는 그래프를 :func:`dgl.graph` 를 사용해서 :class:`~dgl.DGLGraph` 만들고, 그래프 구조를 쿼리하는 API들을 보여준다.

.. figure:: https://data.dgl.ai/asset/image/user_guide_graphch_1.png
    :height: 200px
    :width: 300px
    :align: center

.. code::

    >>> import dgl
    >>> import torch as th

    >>> # edges 0->1, 0->2, 0->3, 1->3
    >>> u, v = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
    >>> g = dgl.graph((u, v))
    >>> print(g) # number of nodes are inferred from the max node IDs in the given edges
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={})

    >>> # Node IDs
    >>> print(g.nodes())
    tensor([0, 1, 2, 3])
    >>> # Edge end nodes
    >>> print(g.edges())
    (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]))
    >>> # Edge end nodes and edge IDs
    >>> print(g.edges(form='all'))
    (tensor([0, 0, 0, 1]), tensor([1, 2, 3, 3]), tensor([0, 1, 2, 3]))

    >>> # If the node with the largest ID is isolated (meaning no edges),
    >>> # then one needs to explicitly set the number of nodes
    >>> g = dgl.graph((u, v), num_nodes=8)

비방향성 그래프를 만들기 위해서는 양방향에 대한 에지들을 만들어야 한다. :func:`dgl.to_bidirected` 함수를 사용하면, 그래프를 양방향의 에지를 갖는 그래프로 변환할 수 있다.

.. code::

    >>> bg = dgl.to_bidirected(g)
    >>> bg.edges()
    (tensor([0, 0, 0, 1, 1, 2, 3, 3]), tensor([1, 2, 3, 0, 3, 0, 0, 1]))

.. note::

    DGL API에서는 일반적으로 텐서 타입이 사용된다. 이는 C 언어에서 효율적으로 저장되는 특징과, 명시적인 데이터 타입, 그리고 디바이스 컨택스트 정보 때문이다. 하지만, 빠른 프로토타입 개발을 지원하기 위해서, 대부분 DGL API는 파이선 iterable (예 list) 및 numpy.array를 함수 인자로 지원하고 있다.

DGL은 노드 및 에지 ID를 저장하는데 :math:`32` 비트 또는 :math:`64` 비트 정수를 사용할 수 있다. 노드와 에지 ID의 데이터 타입은 같아야 한다. :math:`64` 비트를 사용하면 DGL은 노드 또는 에지를 :math:`2^{64} - 1` 개까지 다룰 수 있다. 하지만 그래프의 노드 또는 에지가 :math:`2^{31} - 1` 개 이하인 경우에는 :math:`32` 비트 정수를 사용해야한다. 이유는 속도도 빠르고 저장공간도 적게 사용하기 때문이다. DGL은 이 변환을 위한 방법들을 제공한다. 아래 예제를 보자.

.. code::

    >>> edges = th.tensor([2, 5, 3]), th.tensor([3, 5, 0])  # edges 2->3, 5->5, 3->0
    >>> g64 = dgl.graph(edges)  # DGL uses int64 by default
    >>> print(g64.idtype)
    torch.int64
    >>> g32 = dgl.graph(edges, idtype=th.int32)  # create a int32 graph
    >>> g32.idtype
    torch.int32
    >>> g64_2 = g32.long()  # convert to int64
    >>> g64_2.idtype
    torch.int64
    >>> g32_2 = g64.int()  # convert to int32
    >>> g32_2.idtype
    torch.int32

참고할 API들: :func:`dgl.graph` , :func:`dgl.DGLGraph.nodes` , :func:`dgl.DGLGraph.edges` , :func:`dgl.to_bidirected` ,
:func:`dgl.DGLGraph.int` , :func:`dgl.DGLGraph.long` , 그리고 :py:attr:`dgl.DGLGraph.idtype` 

