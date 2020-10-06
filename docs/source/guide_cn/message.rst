.. _guide_cn-message-passing:

第2章：消息传递范式
================

:ref:`(English Version) guide-message-passing`

Message Passing Paradigm

消息传递范式
----------

Let :math:`x_v\in\mathbb{R}^{d_1}` be the feature for node :math:`v`,
and :math:`w_{e}\in\mathbb{R}^{d_2}` be the feature for edge
:math:`({u}, {v})`. The **message passing paradigm** defines the
following node-wise and edge-wise computation at step :math:`t+1`:

设 :math:`x_v\in\mathbb{R}^{d_1}` 是节点 :math:`v` 的特征， :math:`w_{e}\in\mathbb{R}^{d_2}` 是边 :math:`({u}, {v})` 的特征。
**消息传递范式** 在步骤 :math:`t+1` 定义了以下逐节点和边上的计算：

.. math::  \text{逐边的: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.

.. math::  \text{逐节点的: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).

In the above equations, :math:`\phi` is a **message function**
defined on each edge to generate a message by combining the edge feature
with the features of its incident nodes; :math:`\psi` is an
**update function** defined on each node to update the node feature
by aggregating its incoming messages using the **reduce function**
:math:`\rho`.

在上面的等式中， :math:`\phi` 是定义在每条边上的消息函数，通过将边上特征与其两端节点的特征相结合来生成消息；
:math:`\psi` 是定义在每个节点上的 **更新函数** ，通过使用 **聚合函数** :math:`\rho` 聚合节点接受到的消息来更新节点的特征。

Roadmap

本章路线图
--------

This chapter introduces DGL's message passing APIs, and how to efficiently use them on both nodes and edges.
The last section of it explains how to implement message passing on heterogeneous graphs.

本章首先介绍了DGL的消息传递API。然后讲解了如何高效地在点和边上使用这些API。章节的最后部分解释了如何在异构图上实现消息传递。

* :ref:`guide_cn-message-passing-api`
* :ref:`guide_cn-message-passing-efficient`
* :ref:`guide_cn-message-passing-part`
* :ref:`guide_cn-message-passing-edge`
* :ref:`guide_cn-message-passing-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    message-api
    message-efficient
    message-part
    message-edge
    message-heterograph
