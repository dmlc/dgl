.. _guide_ko-message-passing-edge:

2.4 메시지 전달에 에지 가중치 적용하기
-----------------------------

:ref:`(English Version) <guide-message-passing-edge>`

`GAT <https://arxiv.org/pdf/1710.10903.pdf>`__ 또는 일부 `GCN 변형 <https://arxiv.org/abs/2004.00445>`__ 에서 사용되는 것처럼 메시지 병합이전에 에지의 가중치를 적용하는 것은 GNN 모델링에서 흔하게 사용되는 기법이다. DGL은 이를 다음과 같은 밥벙으로 지원하고 있다.

- 가중치를 에지 피쳐로 저장
- 메시지 함수에서 에지 피쳐를 소스 노드의 피쳐와 곱하기

예를 들면,

.. code::

    import dgl.function as fn

    # Suppose eweight is a tensor of shape (E, *), where E is the number of edges.
    graph.edata['a'] = eweight
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))

이 예제는 eweight를 이제 가중치고 사용하고 있다. 에지 가중치는 보통은 스칼라 값을 갖는다.