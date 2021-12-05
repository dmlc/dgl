.. _guide_ko-graph-heterogeneous:

1.5 이종 그래프 (Heterogeneous Graph)
----------------------------------

:ref:`(English Version)<guide-graph-heterogeneous>`

이종 그래프는 다른 타입의 노드와 에지를 갖는다. 다른 타입의 노드/에지는 독립적인 ID 공간과 피처 저장소를 갖는다. 아래 그램의 예를 보면, user와 game 노드 ID는 모두 0부터 시작하고, 서로 다른 피처들을 갖고 있다.

.. figure:: https://data.dgl.ai/asset/image/user_guide_graphch_2.png

    두 타입의 노드(user와 game)와 두 타입의 에지(follows와 plays)를 갖는 이종 그래프 예

이종 그래프 생성하기
^^^^^^^^^^^^^^^

DGL에서 이종 그래프(짧게 heterograph)는 관계당 하나의 그래프들의 시리즈로 표현된다. 각 관계는 문자열 트리플 ``(source node type, edge type, destination node type)`` 이다. 관계가 에지 타입을 명확하게 하기 때문에, DGL은 이것들을 캐노니컬(canonical) 에지 타입이라고 한다.

아래 코드는 DGL에서 이종 그래프를 만드는 예제이다.

.. code::

    >>> import dgl
    >>> import torch as th

    >>> # Create a heterograph with 3 node types and 3 edges types.
    >>> graph_data = {
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... }
    >>> g = dgl.heterograph(graph_data)
    >>> g.ntypes
    ['disease', 'drug', 'gene']
    >>> g.etypes
    ['interacts', 'interacts', 'treats']
    >>> g.canonical_etypes
    [('drug', 'interacts', 'drug'),
     ('drug', 'interacts', 'gene'),
     ('drug', 'treats', 'disease')]

동종(homogeneous) 및 이분(bipartite) 그래프는 하나의 관계를 갖는 특별한 이종 그래프일 뿐임을 알아두자.

.. code::

    >>> # A homogeneous graph
    >>> dgl.heterograph({('node_type', 'edge_type', 'node_type'): (u, v)})
    >>> # A bipartite graph
    >>> dgl.heterograph({('source_type', 'edge_type', 'destination_type'): (u, v)})

이종 그래프와 연관된 *메타그래프(metagraph)* 는 그래프의 스키마이다. 이것은 노드들과 노드간의 에지들의 집합에 대한 타입 제약 조건을 지정한다. 메타그래프의 노드 :math:`u` 는 연관된 이종 그래프의 노드 타입에 해당한다. 메타그래프의 에지 :math:`(u,v)` 는 연관된 이종 그래프의 노드 타입 :math:`u` 와 노드 타입 :math:`v` 간에 에지가 있다는 것을 알려준다.

.. code::

    >>> g
    Graph(num_nodes={'disease': 3, 'drug': 3, 'gene': 4},
          num_edges={('drug', 'interacts', 'drug'): 2,
                     ('drug', 'interacts', 'gene'): 2,
                     ('drug', 'treats', 'disease'): 1},
          metagraph=[('drug', 'drug', 'interacts'),
                     ('drug', 'gene', 'interacts'),
                     ('drug', 'disease', 'treats')])
    >>> g.metagraph().edges()
    OutMultiEdgeDataView([('drug', 'drug'), ('drug', 'gene'), ('drug', 'disease')])

참고할 API들: :func:`dgl.heterograph` , :py:attr:`~dgl.DGLGraph.ntypes` , :py:attr:`~dgl.DGLGraph.etypes` , :py:attr:`~dgl.DGLGraph.canonical_etypes` , :py:attr:`~dgl.DGLGraph.metagraph`

다양한 타입을 다루기
^^^^^^^^^^^^^^^

노드와 에지가 여러 타입이 사용되는 경우, 타입 관련된 정보를 위한 DGLGraph API를 호출할 때는 노드/에지의 타입을 명시해야한다. 추가로 다른 타입의 노드/에지는 별도의 ID를 갖는다.

.. code::

    >>> # Get the number of all nodes in the graph
    >>> g.num_nodes()
    10
    >>> # Get the number of drug nodes
    >>> g.num_nodes('drug')
    3
    >>> # Nodes of different types have separate IDs,
    >>> # hence not well-defined without a type specified
    >>> g.nodes()
    DGLError: Node type name must be specified if there are more than one node types.
    >>> g.nodes('drug')
    tensor([0, 1, 2])

특정 노드/에지 타입에 대한 피쳐를 설정하고 얻을 때, DGL은 두가지 새로운 형태의 문법을 제공한다 -- `g.nodes['node_type'].data['feat_name']`와 `g.edges['edge_type'].data['feat_name']`.

.. code::

    >>> # Set/get feature 'hv' for nodes of type 'drug'
    >>> g.nodes['drug'].data['hv'] = th.ones(3, 1)
    >>> g.nodes['drug'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])
    >>> # Set/get feature 'he' for edge of type 'treats'
    >>> g.edges['treats'].data['he'] = th.zeros(1, 1)
    >>> g.edges['treats'].data['he']
    tensor([[0.]])

만약 그래프가 오직 한개의 노드/에지 타입을 갖는다면, 노드/에지 타입을 명시할 필요가 없다.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'is similar', 'drug'): (th.tensor([0, 1]), th.tensor([2, 3]))
    ... })
    >>> g.nodes()
    tensor([0, 1, 2, 3])
    >>> # To set/get feature with a single type, no need to use the new syntax
    >>> g.ndata['hv'] = th.ones(4, 1)

.. note::

    에지 타입이 목적지와 도착지 노드의 타입을 고유하게 결정할 수 있다면, 에지 타입을 명시할 때 문자 트리플 대신 한 문자만들 사용할 수 있다. 예를 듬녀, 두 관계 ``('user', 'plays', 'game')`` and ``('user', 'likes', 'game')``를 갖는 이종 그래프가 있을 때, 두 관계를 지정하기 위해서 단지 ``'plays'`` 또는 ``'likes'`` 를 사용해도 된다.

디스크에서 이종 그래프 로딩하기
^^^^^^^^^^^^^^^^^^^^^^^

Comma Separated Values (CSV)
""""""""""""""""""""""""""""

이종 그래프를 저장하는 일반적인 방법은 다른 타입의 노드와 에지를 서로 다른 CSV 파일에 저장하는 것이다. 예를들면 다음과 같다.

.. code::

    # data folder
    data/
    |-- drug.csv        # drug nodes
    |-- gene.csv        # gene nodes
    |-- disease.csv     # disease nodes
    |-- drug-interact-drug.csv  # drug-drug interaction edges
    |-- drug-interact-gene.csv  # drug-gene interaction edges
    |-- drug-treat-disease.csv  # drug-treat-disease edges

동종 그래프의 경우와 동일하게, Pandas와 같은 패키지들을 사용해서 CSV 파일들을 파싱하고, 이를 numpy 배열 또는 프레임워크의 텐서들에 저장하고, 관계 사전을 만들고, 이를 이용해서 이종 그래프를 생성할 수 있다. 이 방법은 GML/JSON과 같은 다른 유명한 포멧들에도 동일하게 적용된다.

DGL 바이너리 포멧
""""""""""""""

DGL은 이종 그래프를 바이너리 포멧으로 저장하고 읽기 위한 함수 :func:`dgl.save_graphs` 와 :func:`dgl.load_graphs` 를 제공한다.

에지 타입 서브그래프
^^^^^^^^^^^^^^^

보존하고 싶은 관계를 명시하고, 피처가 있을 경우는 이를 복사하면서 이종 그래프의 서브그래프를 생성할 수 있다.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... })
    >>> g.nodes['drug'].data['hv'] = th.ones(3, 1)

    >>> # Retain relations ('drug', 'interacts', 'drug') and ('drug', 'treats', 'disease')
    >>> # All nodes for 'drug' and 'disease' will be retained
    >>> eg = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
    ...                                 ('drug', 'treats', 'disease')])
    >>> eg
    Graph(num_nodes={'disease': 3, 'drug': 3},
          num_edges={('drug', 'interacts', 'drug'): 2, ('drug', 'treats', 'disease'): 1},
          metagraph=[('drug', 'drug', 'interacts'), ('drug', 'disease', 'treats')])
    >>> # The associated features will be copied as well
    >>> eg.nodes['drug'].data['hv']
    tensor([[1.],
            [1.],
            [1.]])

이종 그래프를 동종 그래프로 변환하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^

이종 그래프는 다른 타입의 노드/에지와 그것들에 연관된 피쳐들을 관리하는데 깔끔한 인터페이스를 제공한다. 이것을 아래의 경우 특히 유용하다.

1. 다른 타입의 노드/에지에 대한 피쳐가 다른 데이터 타입 또는 크기를 갖는다.
2. 다른 타입의 노드/에지에 다른 연산을 적용하고 싶다.

만약 위 조건을 만족하지 않고 모델링에서 노드/에지 타입의 구별이 필요하지 않는다면, DGL의 :func:`dgl.DGLGraph.to_homogeneous` API를 이용해서 이종 그래프를 동종 그래프로 변환할 수 있다. 이 변환은 다음 절처로 이뤄진다.

1. 모든 타입의 노드/에지를 0부터 시작하는 정수로 레이블을 다시 부여한다.
2. 사용자가 지정한 노드/에지 타입들에 걸쳐서 피쳐들을 합친다.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))})
    >>> g.nodes['drug'].data['hv'] = th.zeros(3, 1)
    >>> g.nodes['disease'].data['hv'] = th.ones(3, 1)
    >>> g.edges['interacts'].data['he'] = th.zeros(2, 1)
    >>> g.edges['treats'].data['he'] = th.zeros(1, 2)

    >>> # By default, it does not merge any features
    >>> hg = dgl.to_homogeneous(g)
    >>> 'hv' in hg.ndata
    False

    >>> # Copy edge features
    >>> # For feature copy, it expects features to have
    >>> # the same size and dtype across node/edge types
    >>> hg = dgl.to_homogeneous(g, edata=['he'])
    DGLError: Cannot concatenate column ‘he’ with shape Scheme(shape=(2,), dtype=torch.float32) and shape Scheme(shape=(1,), dtype=torch.float32)

    >>> # Copy node features
    >>> hg = dgl.to_homogeneous(g, ndata=['hv'])
    >>> hg.ndata['hv']
    tensor([[1.],
            [1.],
            [1.],
            [0.],
            [0.],
            [0.]])

원래의 노드/에지 타입과 타입별 ID들은 :py:attr:`~dgl.DGLGraph.ndata` 와 :py:attr:`~dgl.DGLGraph.edata` 에 저장된다.

.. code::

    >>> # Order of node types in the heterograph
    >>> g.ntypes
    ['disease', 'drug']
    >>> # Original node types
    >>> hg.ndata[dgl.NTYPE]
    tensor([0, 0, 0, 1, 1, 1])
    >>> # Original type-specific node IDs
    >>> hg.ndata[dgl.NID]
    tensor([0, 1, 2, 0, 1, 2])

    >>> # Order of edge types in the heterograph
    >>> g.etypes
    ['interacts', 'treats']
    >>> # Original edge types
    >>> hg.edata[dgl.ETYPE]
    tensor([0, 0, 1])
    >>> # Original type-specific edge IDs
    >>> hg.edata[dgl.EID]
    tensor([0, 1, 0])

모델링 목적으로, 특정 관계들을 모아서 그룹으로 만들고, 그것들에 같은 연산을 적용하고 싶은 경우가 있다. 이를 위해서, 우선 이종 그래프의 에지 타입 서브그래프를 추출하고, 그리고 그 서브그래프를 동종 그래프로 변환한다.

.. code::

    >>> g = dgl.heterograph({
    ...    ('drug', 'interacts', 'drug'): (th.tensor([0, 1]), th.tensor([1, 2])),
    ...    ('drug', 'interacts', 'gene'): (th.tensor([0, 1]), th.tensor([2, 3])),
    ...    ('drug', 'treats', 'disease'): (th.tensor([1]), th.tensor([2]))
    ... })
    >>> sub_g = dgl.edge_type_subgraph(g, [('drug', 'interacts', 'drug'),
    ...                                    ('drug', 'interacts', 'gene')])
    >>> h_sub_g = dgl.to_homogeneous(sub_g)
    >>> h_sub_g
    Graph(num_nodes=7, num_edges=4,
          ...)
