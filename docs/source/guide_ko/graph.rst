.. _guide_ko-graph:

1장: 그래프
=========

:ref:`(English version)<guide-graph>`

그래프는 앤티티들(entity 또는 노드들)과 노드들간의 관계(에지)로 표현되며, 노드와 에지들을 타입을 갖을 수 있다. (예를 들어, ``"user"`` 와 ``"item"`` 은 서로 다른 타입의 노드들이다.) DGL은 :class:`~dgl.DGLGraph` 를 핵심 자료 구조로 갖는 그래프-중심의 프로그래밍 추상화를 제공한다. :class:`~dgl.DGLGraph` 그래프의 구조, 그 그래프의 노드 및 에지 피처들과 이 컴포넌트들을 사용해서 수행된 연산 결과를 다루는데 필요한 인터페이스를 제공한다.

로드맵
-------

이 장은 1.1절의 그래프 정의에 대한 간단한 소개를 시작으로 :class:`~dgl.DGLGraph`: 의 몇가지 핵심 개념을 소개한다.

* :ref:`guide_ko-graph-basic`
* :ref:`guide_ko-graph-graphs-nodes-edges`
* :ref:`guide_ko-graph-feature`
* :ref:`guide_ko-graph-external`
* :ref:`guide_ko-graph-heterogeneous`
* :ref:`guide_ko-graph-gpu`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    graph-basic
    graph-graphs-nodes-edges
    graph-feature
    graph-external
    graph-heterogeneous
    graph-gpu
