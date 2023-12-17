.. _guide-minibatch:

Chapter 6: Stochastic Training on Large Graphs
=======================================================

:ref:`(中文版) <guide_cn-minibatch>`

If we have a massive graph with, say, millions or even billions of nodes
or edges, usually full-graph training as described in
:ref:`guide-training`
would not work. Consider an :math:`L`-layer graph convolutional network
with hidden state size :math:`H` running on an :math:`N`-node graph.
Storing the intermediate hidden states requires :math:`O(NLH)` memory,
easily exceeding one GPU’s capacity with large :math:`N`.

This section provides a way to perform stochastic minibatch training,
where we do not have to fit the feature of all the nodes into GPU.

Overview of Neighborhood Sampling Approaches
--------------------------------------------

Neighborhood sampling methods generally work as the following. For each
gradient descent step, we select a minibatch of nodes whose final
representations at the :math:`L`-th layer are to be computed. We then
take all or some of their neighbors at the :math:`L-1` layer. This
process continues until we reach the input. This iterative process
builds the dependency graph starting from the output and working
backwards to the input, as the figure below shows:

.. figure:: https://data.dgl.ai/asset/image/guide_6_0_0.png
   :alt: Imgur



With this, one can save the workload and computation resources for
training a GNN on a large graph.

DGL provides a few neighborhood samplers and a pipeline for training a
GNN with neighborhood sampling, as well as ways to customize your
sampling strategies.

Roadmap
-----------

The chapter starts with sections for training GNNs stochastically under
different scenarios.

* :ref:`guide-minibatch-node-classification-sampler`
* :ref:`guide-minibatch-edge-classification-sampler`
* :ref:`guide-minibatch-link-classification-sampler`

The remaining sections cover more advanced topics, suitable for those who
wish to develop new sampling algorithms, new GNN modules compatible with
mini-batch training and understand how evaluation and inference can be
conducted in mini-batches.

* :ref:`guide-minibatch-customizing-neighborhood-sampler`
* :ref:`guide-minibatch-sparse`
* :ref:`guide-minibatch-custom-gnn-module`
* :ref:`guide-minibatch-inference`

The following are performance tips for implementing and using neighborhood
sampling:

* :ref:`guide-minibatch-gpu-sampling`
* :ref:`guide-minibatch-parallelism`

.. toctree::
    :maxdepth: 1
    :hidden:
    :glob:

    minibatch-node
    minibatch-edge
    minibatch-link
    minibatch-custom-sampler
    minibatch-sparse
    minibatch-nn
    minibatch-inference
    minibatch-gpu-sampling
    minibatch-parallelism
