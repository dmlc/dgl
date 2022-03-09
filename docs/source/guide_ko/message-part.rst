.. _guide_ko-message-passing-part:

2.3 그래프 일부에 메지시 전달 적용하기
------------------------------

:ref:`(English Version) <guide-message-passing-part>`

그래프 노드의 일부만 업데이트를 하기 원하는 경우, 업데이트를 하고 싶은 노드들의 ID를 사용해서 서브그래프를 만든 후, 그 서브그래프에 :meth:`~dgl.DGLGraph.update_all` 를 호출하는 방법으로 가능하다.

.. code::

    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func)

이는 미니-배치 학습에서 흔히 사용되는 방법이다. 자세한 사용법은 :ref:`guide_ko-minibatch` 참고하자.