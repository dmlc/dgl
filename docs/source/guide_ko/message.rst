.. _guide_ko-message-passing:

2장: 메지시 전달(Message Passing)
=============================

:ref:`(English Version) <guide-message-passing>`

메지시 전달 패러다임(Message Passing Paradigm)
-----------------------------------------

:math:`x_v\in\mathbb{R}^{d_1}` 이 노드 :math:`v` 의 피처이고, :math:`w_{e}\in\mathbb{R}^{d_2}` 가 에지 :math:`({u}, {v})` 의 피처라고 하자. **메시지 전달 패러다임** 은 :math:`t+1` 단계에서 노드별(node-wise) 그리고 에지별(edge-wise)의 연산을 다음과 같이 정의한다:

.. math::  \text{에지별: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.

.. math::  \text{노드별: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).

위 수식에서 :math:`\phi` 는 각 에지에 대한 **메시지 함수** 로서 에지의 부속 노드(incident node)들의 피처를 그 에지 피처와 합쳐서 메시지를 만드는 역할을 수행한다. :math:`\psi` 는 각 노드에 대한 **업데이트 함수** 로, **축소 함수(reduce function)** :math:`\rho` 를 사용해서 전달된 메시지들을 통합하는 방식으로 노드의 피처를 업데이트한다.

로드맵
----

이 장는 DGL의 메시지 전달 API들과, 노드와 에지에 효율적으로 적용하는 방법을 소개한다. 마지막 절에서는 이종 그래프에 메시지 전달을 어떻게 구현하는지 설명한다.

* :ref:`guide_ko-message-passing-api`
* :ref:`guide_ko-message-passing-efficient`
* :ref:`guide_ko-message-passing-part`
* :ref:`guide_ko-message-passing-edge`
* :ref:`guide_ko-message-passing-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    message-api
    message-efficient
    message-part
    message-edge
    message-heterograph
