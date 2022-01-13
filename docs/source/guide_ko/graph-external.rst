.. _guide_ko-graph-external:

1.4 외부 소스를 사용한 그래프 생성하기
-----------------------------------------

:ref:`(English Version)<guide-graph-external>`

외부 소스들로부터 :class:`~dgl.DGLGraph` 를 만드는 옵션들:

- 그래프 및 회소 행렬을 위한 python 라이브러리(NetworkX 및 SciPy)로부터 변환하기
- 디스크에서 그래프를 로딩하기

이 절에서는 다른 그래프를 변환해서 그래프를 생성하는 함수들은 다루지 않겠다. 그 방법들에 대한 소개는 매뉴얼의 API를 참조하자.

외부 라이브러리를 사용해서 그래프 생성하기
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

아래 코드는 SciPy 희소행렬과 NetworkX 그래프로부터 그래프를 생성하는 예제이다.

.. code::

    >>> import dgl
    >>> import torch as th
    >>> import scipy.sparse as sp
    >>> spmat = sp.rand(100, 100, density=0.05) # 5% nonzero entries
    >>> dgl.from_scipy(spmat)                   # from SciPy
    Graph(num_nodes=100, num_edges=500,
          ndata_schemes={}
          edata_schemes={})

    >>> import networkx as nx
    >>> nx_g = nx.path_graph(5) # a chain 0-1-2-3-4
    >>> dgl.from_networkx(nx_g) # from networkx
    Graph(num_nodes=5, num_edges=8,
          ndata_schemes={}
          edata_schemes={})

`nx.path_graph(5)` 로부터 만들면 생성된 :class:`~dgl.DGLGraph` 는 4개가 아니라 8개의 에지를 갖는 점을 유의하자. 이유는 `nx.path_graph(5)` 는 방향이 없는 NetworkX 그래프 :class:`networkx.Graph` 를 만드는데, :class:`~dgl.DGLGraph` 는 항상 방향이 있는 그래프이기 때문이다. 방향이 없는 NetworkX 그래프를 :class:`~dgl.DGLGraph` 로 변환하면, DGL은 내부적으로 방향이 없는 에지를 두개의 방향이 있는 에지로 변환한다. :class:`networkx.DiGraph` 를 사용하면 이런 현상을 피할 수 있다.

.. code::

    >>> nxg = nx.DiGraph([(2, 1), (1, 2), (2, 3), (0, 0)])
    >>> dgl.from_networkx(nxg)
    Graph(num_nodes=4, num_edges=4,
          ndata_schemes={}
          edata_schemes={})

.. note::

    내부적으로 DGL은 SciPy 행렬과 NetworkX 그래프를 텐서로 변환해서 그래프를 만든다. 따라서, 이 생성 방법은 성능이 중요한 곳에 사용되기 적합하지 않다.

참고할 API들: :func:`dgl.from_scipy` , :func:`dgl.from_networkx` .

디스크에서 그래프 로딩하기
^^^^^^^^^^^^^^^^^^^

그래프를 저장하기 위한 여러 데이터 포멧들이 있는데, 모든 옵션들을 나열하기는 불가능하다. 그래서 이 절에서는 공통적인 것들에 대한 일반적인 참조만 소개한다.

Comma Separated Values (CSV)
""""""""""""""""""""""""""""

아주 일반적인 포멧으로 CSV가 사용된다. 이는 노드, 에치, 그리고 그것들의 피처들을 테이블 형태로 저장한다.

.. table:: nodes.csv

   +-----------+
   |age, title |
   +===========+
   |43, 1      |
   +-----------+
   |23, 3      |
   +-----------+
   |...        |
   +-----------+

.. table:: edges.csv

   +-----------------+
   |src, dst, weight |
   +=================+
   |0, 1, 0.4        |
   +-----------------+
   |0, 3, 0.9        |
   +-----------------+
   |...              |
   +-----------------+

잘 알려진 Python 라이브러리들(예, pandas)을 사용해서 이 형태의 데이터를 python 객체(예, :class:`numpy.ndarray` )로 로딩하고, 이를 DGLGraph로 변환하는데 사용할 수 있다. 만약 백엔드 프레임워크가 디스크에서 텐서를 저장하고/읽는 기능(예, :func:`torch.save` , :func:`torch.load` )을 제공한다면, 그래프를 만드는데 이용할 수 있다.

함께 참조하기: `Tutorial for loading a Karate Club Network from edge pairs CSV <https://github.com/dglai/WWW20-Hands-on-Tutorial/blob/master/basic_tasks/1_load_data.ipynb>`_.

JSON/GML 포멧
""""""""""""

특별히 빠르지는 않지만 NetworkX는 `다양한 데이터 포멧 <https://networkx.github.io/documentation/stable/reference/readwrite/index.html>`_ 을 파싱하는 유틸리티들을 제공하는데, 이를 통해서 DGL 그래프를 만들 수 있다.

DGL 바이너리 포멧
""""""""""""""

DGL은 디스크에 그래프를 바이너리 형태로 저장하고 로딩하는 API들을 제공한다. 그래프 구조와 더불어, API들은 피처 데이터와 그래프 수준의 레이블 데이터도 다룰 수 있다. DGL은 그래프를 직접 S3 또는 HDFS에 체크포인트를 할 수 있는 기능을 제공한다. 러퍼런스 메뉴얼에 자세한 내용이 있으니 참고하자.

참고할 API들: :func:`dgl.save_graphs` , :func:`dgl.load_graphs`
