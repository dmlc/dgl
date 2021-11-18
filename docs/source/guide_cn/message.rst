.. _guide_cn-message-passing:

第2章：消息传递范式
===========================

:ref:`(English Version) <guide-message-passing>`

消息传递是实现GNN的一种通用框架和编程范式。它从聚合与更新的角度归纳总结了多种GNN模型的实现。

消息传递范式
----------------------

假设节点 :math:`v` 上的的特征为 :math:`x_v\in\mathbb{R}^{d_1}`，边 :math:`({u}, {v})` 上的特征为 :math:`w_{e}\in\mathbb{R}^{d_2}`。
**消息传递范式** 定义了以下逐节点和边上的计算：

.. math::  \text{边上计算: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.

.. math::  \text{点上计算: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).

在上面的等式中， :math:`\phi` 是定义在每条边上的消息函数，它通过将边上特征与其两端节点的特征相结合来生成消息。
**聚合函数** :math:`\rho` 会聚合节点接受到的消息。 **更新函数** :math:`\psi` 会结合聚合后的消息和节点本身的特征来更新节点的特征。

本章路线图
--------------------

本章首先介绍了DGL的消息传递API。然后讲解了如何高效地在点和边上使用这些API。本章的最后一节解释了如何在异构图上实现消息传递。

* :ref:`guide_cn-message-passing-api`
* :ref:`guide_cn-message-passing-efficient`
* :ref:`guide_cn-message-passing-part`
* :ref:`guide_cn-message-passing-heterograph`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    message-api
    message-efficient
    message-part
    message-heterograph
