.. _guide_ko-graph-feature:

1.3 노드와 에지의 피처
--------------------------

:ref:`(English Version)<guide-graph-feature>`

노드들과 에지들의 그래프별 속성을 저장하기 위해서, :class:`~dgl.DGLGraph` 의 노드들과 에지들은 이름을 갖는 사용자 정의 피쳐를 갖을 수 있다. :py:attr:`~dgl.DGLGraph.ndata` 와 :py:attr:`~dgl.DGLGraph.edata` 인터페이스를 이용해서 이 피쳐들을 접근할 수 있다. 예를 들어, 아래 코드는 두 노드에 대한 피쳐를 생성하고(라인 8과 15에서 ``'x'`` 와 ``'y'`` 이름 피처), 한개의 에지 피처(라인 9에서 ``'x'`` 이름 피처)를 생성한다.

.. code-block:: python
    :linenos:

    >>> import dgl
    >>> import torch as th
    >>> g = dgl.graph(([0, 0, 1, 5], [1, 2, 2, 0])) # 6 nodes, 4 edges
    >>> g
    Graph(num_nodes=6, num_edges=4,
          ndata_schemes={}
          edata_schemes={})
    >>> g.ndata['x'] = th.ones(g.num_nodes(), 3)               # node feature of length 3
    >>> g.edata['x'] = th.ones(g.num_edges(), dtype=th.int32)  # scalar integer feature
    >>> g
    Graph(num_nodes=6, num_edges=4,
          ndata_schemes={'x' : Scheme(shape=(3,), dtype=torch.float32)}
          edata_schemes={'x' : Scheme(shape=(,), dtype=torch.int32)})
    >>> # different names can have different shapes
    >>> g.ndata['y'] = th.randn(g.num_nodes(), 5)
    >>> g.ndata['x'][1]                  # get node 1's feature
    tensor([1., 1., 1.])
    >>> g.edata['x'][th.tensor([0, 3])]  # get features of edge 0 and 3
        tensor([1, 1], dtype=torch.int32)

:py:attr:`~dgl.DGLGraph.ndata`/:py:attr:`~dgl.DGLGraph.edata` 인터페이스의 중요한 사실들:

- 숫자 타입(예, float, double, int)의 피처들만 허용된다. 피처는 스칼라, 벡터, 또는 다차원 텐서가 가능하다.
- 각 노드 피처는 고유한 이름을 갖고, 각 에지 피쳐도 고유한 이름을 갖는다. 노드와 에지의 피쳐는 같은 이름을 갖을 수 있다. (예, 위 예의 'x')
- 턴서 할당으로 피처가 만들어진다. 즉, 피처를 그래프의 각 노드/에지에 할당하는 것이다. 텐서의 첫번째 차원은 그래프의 노드/에지들의 개수와 같아야 한다. 그래프의 노드/에지의 일부에만 피쳐를 할당하는 것은 불가능하다.
- 같은 이름의 피처들은 같은 차원 및 같은 타입을 갖아야 한다.
- 피처 텐서는 행 위주(row-major)의 레이아웃을 따른다. 각 행-슬라이스는 한 노드 또는 이제의 피처를 저장한다. (아래 예제의 16줄 및 18줄을 보자)

가중치 그래프인 경우, 에지 피처로 가중치를 저장할 수 있다.

.. code-block:: python

    >>> # edges 0->1, 0->2, 0->3, 1->3
    >>> edges = th.tensor([0, 0, 0, 1]), th.tensor([1, 2, 3, 3])
    >>> weights = th.tensor([0.1, 0.6, 0.9, 0.7])  # weight of each edge
    >>> g = dgl.graph(edges)
    >>> g.edata['w'] = weights  # give it a name 'w'
    >>> g
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={'w' : Scheme(shape=(,), dtype=torch.float32)})

참고할 API들: :py:attr:`~dgl.DGLGraph.ndata` , :py:attr:`~dgl.DGLGraph.edata`
