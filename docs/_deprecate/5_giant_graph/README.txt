.. _tutorials5-index:


Training on giant graphs
=============================

* **Sampling** `[paper] <https://arxiv.org/abs/1710.10568>`__ `[tutorial]
  <5_giant_graph/1_sampling_mx.html>`__ `[MXNet code]
  <https://github.com/dmlc/dgl/tree/master/examples/mxnet/sampling>`__ `[Pytorch code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/sampling>`__:
  You can perform neighbor sampling and control-variate sampling to train a
  graph convolution network and its variants on a giant graph.
* **Scale to giant graphs** `[tutorial] <5_giant_graph/2_giant.html>`__
  `[MXNet code] <https://github.com/dmlc/dgl/tree/master/examples/mxnet/sampling>`__
  `[Pytorch code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/sampling>`__:
  You can find two components (graph store and distributed sampler) to scale to
  graphs with hundreds of millions of nodes.
