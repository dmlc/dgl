.. _guide-message-passing:

Chapter 2: Message Passing
==========================

:ref:`(中文版) <guide_cn-message-passing>`

Message Passing Paradigm
------------------------

Let :math:`x_v\in\mathbb{R}^{d_1}` be the feature for node :math:`v`,
and :math:`w_{e}\in\mathbb{R}^{d_2}` be the feature for edge
:math:`({u}, {v})`. The **message passing paradigm** defines the
following node-wise and edge-wise computation at step :math:`t+1`:

.. math::  \text{Edge-wise: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.

.. math::  \text{Node-wise: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).

In the above equations, :math:`\phi` is a **message function**
defined on each edge to generate a message by combining the edge feature
with the features of its incident nodes; :math:`\psi` is an
**update function** defined on each node to update the node feature
by aggregating its incoming messages using the **reduce function**
:math:`\rho`.

Roadmap
-------

This chapter introduces DGL's message passing APIs, and how to efficiently use them on both nodes and edges.
The last section of it explains how to implement message passing on heterogeneous graphs.

* :ref:`guide-message-passing-api`
* :ref:`guide-message-passing-efficient`
* :ref:`guide-message-passing-part`
* :ref:`guide-message-passing-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    message-api
    message-efficient
    message-part
    message-heterograph
