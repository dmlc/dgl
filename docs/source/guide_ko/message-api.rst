.. _guide_ko-message-passing-api:

2.1 빌트인 함수 및 메시지 전달 API들
-----------------------------

:ref:`(English Version) <guide-message-passing-api>`

DGL에서 **메시지 함수** 는 한개의 인자 ``edges`` 를 갖는데, 이는 :class:`~dgl.udf.EdgeBatch` 의 객체이다. 메시지 전달이 실행되는 동안 DGL은 에지 배치를 표현하기 위해서 이 객체를 내부적으로 생성한다. 이것은 3개의 맴버, ``src`` , ``dst`` , 그리고 ``data`` 를 갖고, 이는 각각 소스 노드, 목적지 노드, 그리고 에지의 피쳐를 의미한다.

**축약 함수(reduce function)** 는 한개의 인자 ``nodes`` 를 갖는데, 이는 :class:`~dgl.udf.NodeBatch` 의 객체이다. 메시지 전달이 실행되는 동안 DGL은 노드 배치를 표현하기 위해서 이 객체를 내부적으로 생성한다. 이 객체는 ``mailbox`` 라는 맴버를 갖는데, 이는 배치에 속한 노드들에게 전달된 메시지들을 접근 방법을 제공한다. 가장 흔한 축약 함수로는 ``sum`` , ``max`` , ``min`` 등이 있다.

**업데이트 함수** 는 위에서 언급한 ``nodes`` 를 한개의 인자로 갖는다. 이 함수는 ``축약 함수`` 의 집계 결과에 적용되는데, 보통은 마지막 스탭에서 노드의 원래 피처와 이 결과와 결합하고, 그 결과를 노드의 피처로 저장한다.

DGL은 일반적으로 사용되는 메시지 전달 함수들과 축약 함수들을 ``dgl.function`` 네임스패이스에 **빌트인** 으로 구현하고 있다. 일반적으로, **가능한 경우라면 항상** DLG의 빌트인 함수를 사용하는 것을 권장하는데, 그 이유는 이 함수들은 가장 최적화된 형태로 구현되어 있고, 차원 브로드캐스팅을 자동으로 해주기 때문이다.

만약 여러분의 메시지 전달 함수가 빌트인 함수로 구현이 불가능하다면, 사용자 정의 메시지/축소 함수를 직접 구현할 수 있다. 이를 **UDF** 라고 한다.

빌트인 메시지 함수들은 단항(unary) 또는 이상(binary)이다. 단항의 경우 DGL은 ``copy`` 를 지원한다. 이항 함수로 DGL은 ``add`` , ``sub`` , ``mul`` , ``div`` , 그리고 ``dot`` 를 지원한다. 빌트인 메시지 함수의 이름 규칙은 다음과 같다. ``u`` 는 ``src`` 노드를, ``v`` 는 ``dst`` 노드를 그리고 ``e`` 는 ``edges`` 를 의미한다. 이 함수들에 대한 파라미터들은 관련된 노드와 에지의 입력과 출력 필드 이름을 지칭하는 문자열이다. 지원되는 빌트인 함수의 목록은 :ref:`api-built-in` 을 참고하자. 한가지 예를 들면, 소스 노드의 ``hu`` 피처와 목적지 노드의 ``hv`` 피처를 더해서 그 결과를 에지의 ``he`` 필드에 저장하는 것을 빌트인 함수 ``dgl.function.u_add_v('hu', 'hv', 'he')`` 를 사용해서 구현할 수 있다. 이와 동일한 기능을 하는 메시지 UDF는 다음과 같다.

.. code::

    def message_func(edges):
         return {'he': edges.src['hu'] + edges.dst['hv']}

빌트인 축약 함수는 ``sum``, ``max``, ``min`` 그리고 ``mean`` 연산을 지원한다. 보통 축약 함수는 두개의 파라메터를 갖는데, 하나는 ``mailbox`` 의 필드 이름이고, 다른 하나는 노드 피처의 필드 이름이다. 이는 모두 문자열이다. 예를 들어, ``dgl.function.sum('m', 'h')`` 는 메시지 ``m`` 을 합하는 아래 축약 UDF와 같다.

.. code::

    import torch
    def reduce_func(nodes):
         return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

UDF의 고급 사용법을 더 알고 싶으면 :ref:`apiudf` 를 참고하자.

:meth:`~dgl.DGLGraph.apply_edges` 를 사용해서 메시지 전달 함수를 호출하지 않고 에지별 연산만 호출하는 것도 가능하다. :meth:`~dgl.DGLGraph.apply_edges` 는 파라미터로 메시지 함수를 받는데, 기본 설정으로는 모든 에지의 피쳐를 업데이트한다. 다음 예를 살펴보자.

.. code::

    import dgl.function as fn
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

메시지 전달을 위한 :meth:`~dgl.DGLGraph.update_all` 는 하이레벨 API로 메시지 생성, 메시지 병합 그리고 노드 업데이트를 단일 호출로 합쳤는데, 전반적으로 최적화할 여지가 남아있다.

:meth:`~dgl.DGLGraph.update_all` 의 파라메터들은 메시지 함수, 축약 함수, 그리고 업데이트 함수이다. :meth:`~dgl.DGLGraph.update_all` 를 호출할 때 업데이트 함수를 지정하지 않는 경우, 업데이트 함수는 ``update_all`` 밖에서 수행될 수 있다. DGL은 이 방법을 권장하는데, 업데이트 함수는 코드를 간결하게 만들기 위해서 보통은 순수 텐서 연산으로 구현되어 있기 때문이다. 예를 들면, 다음과 같다.

.. code::

    def update_all_example(graph):
        # store the result in graph.ndata['ft']
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        # Call update function outside of update_all
        final_ft = graph.ndata['ft'] * 2
        return final_ft

이 함수는 소스 노드의 피처 ``ft`` 와 에지 피처 ``a`` 를 곱해서 메시지 ``m`` 을 생성하고, 메시지 ``m`` 들을 더해서 노드 피처 ``ft`` 를 업데이트하고, 마지막으로 ``final_ft`` 결과를 구하기 위해서 ``ft`` 에 2를 곱하고 있다. 호출이 완료되면 DGL은 중간에 사용된 메시지들 ``m`` 을 제거한다. 위 함수를 수학 공식으로 표현하면 다음과 같다.

.. math::  {final\_ft}_i = 2 * \sum_{j\in\mathcal{N}(i)} ({ft}_j * a_{ij})

DGL의 빌트인 함수는 부동소수점 데이터 타입을 지원한다. 즉, 피쳐들은 반드시 ``half`` (``float16``), ``float``, 또는 ``double`` 텐서여야만 한다. ``float16`` 데이터 타입에 대한 지원은 기본 설정에서는 비활성화되어 있다. 그 이유는 이를 지원하기 위해서는 ``sm_53`` (Pascal, Volta, Turing, 그리고 Ampere 아키텍타)와 같은 최소한의 GPU 컴퓨팅 능력이 요구되기 때문이다. 

사용자는 DGL 소스 컴파일을 통해서 mixed precision training을 위해서 float16을 활성화시킬 수 있다. (자세한 내용은 :doc:`Mixed Precision Training <mixed_precision>` 튜토리얼 참고)
