"""
Introduction of Neighbor Sampling for GNN Training
==================================================

In :doc:`previous tutorials <1_introduction>` you have learned how to
train GNNs by computing the representations of all nodes on a graph.
However, sometimes your graph is too large to fit the computation of all
nodes in a single GPU.

By the end of this tutorial, you will be able to

-  Understand the pipeline of stochastic GNN training.
-  Understand what is neighbor sampling and why it yields a bipartite
   graph for each GNN layer.
"""


######################################################################
# Message Passing Review
# ----------------------
# 
# Recall that in `Gilmer et al. <https://arxiv.org/abs/1704.01212>`__
# (also in :doc:`message passing tutorial <3_message_passing>`), the
# message passing formulation is as follows:
# 
# .. math::
# 
# 
#    m_{u\to v}^{(l)} = M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\to v}^{(l-1)}\right)
# 
# .. math::
# 
# 
#    m_{v}^{(l)} = \sum_{u\in\mathcal{N}(v)}m_{u\to v}^{(l)}
# 
# .. math::
# 
# 
#    h_v^{(l)} = U^{(l)}\left(h_v^{(l-1)}, m_v^{(l)}\right)
# 
# where DGL calls :math:`M^{(l)}` the *message function*, :math:`\sum` the
# *reduce function* and :math:`U^{(l)}` the *update function*. Note that
# :math:`\sum` here can represent any function and is not necessarily a
# summation.
# 
# Essentially, the :math:`l`-th layer representation of a single node
# depends on the :math:`(l-1)`-th layer representation of the same node,
# as well as the :math:`(l-1)`-th layer representation of the neighboring
# nodes. Those :math:`(l-1)`-th layer representations then depend on the
# :math:`(l-2)`-th layer representation of those nodes, as well as their
# neighbors.
# 
# The following animation shows how a 2-layer GNN is supposed to compute
# the output of node 5:
# 
# |image1|
# 
# You can see that to compute node 5 from the second layer, you will need
# its direct neighbors’ first layer representations (colored in yellow),
# which in turn needs their direct neighbors’ (i.e. node 5’s second-hop
# neighbors’) representations (colored in green).
# 
# .. |image1| image:: https://data.dgl.ai/tutorial/img/sampling.gif
# 


######################################################################
# Neighbor Sampling Overview
# --------------------------
# 
# You can also see from the previous example that computing representation
# for a small number of nodes often requires input features of a
# significantly larger number of nodes. Taking all neighbors for message
# aggregation is often too costly since the nodes needed for input
# features would easily cover a large portion of the graph, especially for
# real-world graphs which are often
# `scale-free <https://en.wikipedia.org/wiki/Scale-free_network>`__.
# 
# Neighbor sampling addresses this issue by selecting a subset of the
# neighbors to perform aggregation. For instance, to compute
# :math:`\boldsymbol{h}_8^{(2)}`, you can choose two of the neighbors
# instead of all of them to aggregate, as in the following animation:
# 
# |image2|
# 
# You can see that this method could give us fewer nodes needed for input
# features.
# 
# .. |image2| image:: https://data.dgl.ai/tutorial/img/bipartite.gif
# 


######################################################################
# You can also notice in the animation above that the computation
# dependencies can be described as a series of bipartite graphs. The
# output nodes are on one side and all the nodes necessary for inputs are
# on the other side. The arrows indicate how the sampled neighbors
# propagates messages to the nodes. Note that the output nodes themselves
# are also included in the input nodes, because their own input features
# are also necessary in the computation.
# 


######################################################################
# What’s next?
# ------------
# 
# :doc:`Stochastic GNN Training for Node Classification in
# DGL <L1_large_node_classification>`
# 

