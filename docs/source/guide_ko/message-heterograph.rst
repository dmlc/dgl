.. _guide_ko-message-passing-heterograph:

2.5 이종 그래프에서의 메시지 전달
--------------------------

:ref:`(English Version) <guide-message-passing-heterograph>`

이종 그래프 ( :ref:`guide_ko-graph-heterogeneous` ) 또는 헤테로그래프는 여러 타입의 노드와 에지를 갖는 그래프이다. 각 노드와 에지의 특징을 표현하기 위해서 다른 타입의 속성을 갖기 위해서 노드와 에지들이 다른 타입을 갖을 수 있다. 복잡한 그래프 뉴럴 네트워크들에서 어떤 노드나 에지 타입들은 다른 차원들을 갖게 모델링 되기도 한다.

이종 그래프에서 메시지 전달은 두 파트로 나뉜다:

1. 각 관계(relation) r에 대한, 메지시 연산과 집계(aggregation)
2. 가 노트 타입에 대한 모든 관계의 집계 결과를 합치는 축약(reduction)

이종 그래프에서 메시지 전달을 담당하는 DGL 인터페이스는 :meth:`~dgl.DGLGraph.multi_update_all` 이다. :meth:`~dgl.DGLGraph.multi_update_all` 는 :meth:`~dgl.DGLGraph.update_all` 에 대한 파라메터들을 갖는 사전(dictionary)을 인자로 받는다. 이 사전의 각 키값는 관계이고, 그에 대한 값은 크로스 타입 리듀셔(cross type reducer)에 대한 문자열이다. Reducer는 ``sum``, ``min``, ``max``, ``mean``, ``stack`` 중에 하나가 된다. 예제는 다음과 같다.

.. code::

    import dgl.function as fn

    for c_etype in G.canonical_etypes:
        srctype, etype, dsttype = c_etype
        Wh = self.weight[etype](feat_dict[srctype])
        # Save it in graph for message passing
        G.nodes[srctype].data['Wh_%s' % etype] = Wh
        # Specify per-relation message passing functions: (message_func, reduce_func).
        # Note that the results are saved to the same destination feature 'h', which
        # hints the type wise reducer for aggregation.
        funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
    # Trigger message passing of multiple types.
    G.multi_update_all(funcs, 'sum')
    # return the updated node feature dictionary
    return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
