.. _guide_ko-message-passing-efficient:

2.2 효율적인 메시지 전달 코드 작성 방법
------------------------------

:ref:`(English Version) <guide-message-passing-efficient>`

DGL은 메시지 전달에 대한 메모리 사용과 연산 속드를 최적화하고 있다. 이 최적화들을 활용하는 일반적으로 사용되는 방법은 직접 메시지 전달 함수를 만들어서 이를 :meth:`~dgl.DGLGraph.update_all` 호출시 빌트인 함수와 함께 파라메터로 사용하는 것이다. 

만약 그래프의 에지들의 수가 노드들의 수보다 훨씬 많은 경우에는 노드에서 에지로의 불필요한 메모리 복사를 피하는 것이 도움이 된다. 에지에 메시지를 저장할 필요가 있는 :class:`~dgl.nn.pytorch.conv.GATConv` 와 같은 경우에는 빌트인 함수를 사용해서 :meth:`~dgl.DGLGraph.apply_edges` 를 호출해야 한다. 때로는 에지에 저장할 메시지의 차원이 너무 커서 메모리를 많이 차지하기도 한다. DGL에서는 가능한 에지 피쳐의 차원을 낮추는 것을 권장한다.

에지에 대한 연산을 노드로 분할하여 이를 달성하는 방법에 대한 예제이다. 이 방법은 다음과 같다. ``src`` 피쳐와 ``dst`` 피쳐를 연결하고, 선형 레이어 :math:`W\times (u || v)`를 적용하는 경우를 들어보자. ``src``와 ``dst`` 피처 차원은 매우 높은 반면에 선형 레이어의 결과 차원은 낮다고 가정하자. 이 예제를 직관적으로 구현하면 다음과 같다.

.. code::

    import torch
    import torch.nn as nn

    linear = nn.Parameter(torch.FloatTensor(size=(node_feat_dim * 2, out_dim)))
    def concat_message_function(edges):
         return {'cat_feat': torch.cat([edges.src['feat'], edges.dst['feat']], dim=1)}
    g.apply_edges(concat_message_function)
    g.edata['out'] = g.edata['cat_feat'] @ linear

제안하는 구현은 이 선형 연산을 두개로 나누는 것이다. 하나는 ``src`` 피처에 적용하고, 다른 하나는 ``dst`` 피쳐에 적용한다. 그 후, 에지에 대한 두 선형 연산의 결과를 마지막 단계에서 더한다. 즉, :math:`W_l\times u + W_r \times v` 를 실행하는 것이다. :math:`W` 행렬의 왼쪽 반과 오른쪽 반이 각각 :math:`W_l` 와 :math:`W_r` 일 때, :math:`W \times (u||v) = W_l \times u + W_r \times v` 가 성립하기 때문에 가능하다.

.. code::

    import dgl.function as fn

    linear_src = nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
    linear_dst = nn.Parameter(torch.FloatTensor(size=(node_feat_dim, out_dim)))
    out_src = g.ndata['feat'] @ linear_src
    out_dst = g.ndata['feat'] @ linear_dst
    g.srcdata.update({'out_src': out_src})
    g.dstdata.update({'out_dst': out_dst})
    g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))

위 두 구현은 수학적으로 동일하다. 후자가 더 효율적인데, 그 이유는 메모리 비효율적인 에지에 feat_src와 feat_dst의 저장이 필요가 없기 때문이다. 추가로, 합은 연산속도가 더 빠르고 메모리 사용량을 줄인 DGL의 빌트인 함수 ``u_add_v`` 를 사용하면 최적화될 수 있다. 