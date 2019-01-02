.. _tutorials3-index:

Generative models
==================

* **DGMG** `[paper] <https://arxiv.org/abs/1803.03324>`__ `[tutorial]
  <3_generative_model/5_dgmg.html>`__ `[code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg>`__:
  this model belongs to the important family that deals with structural
  generation. DGMG is interesting because its state-machine approach is the
  most general. It is also very challenging because, unlike Tree-LSTM, every
  sample has a dynamic, probability-driven structure that is not available
  before training. We are able to progressively leverage intra- and
  inter-graph parallelism to steadily improve the performance.

* **JTNN** `[paper] <https://arxiv.org/abs/1802.04364>`__ `[code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/jtnn>`__:
  unlike DGMG, this paper generates molecular graphs using the framework of
  variational auto-encoder. Perhaps more interesting is its approach to build
  structure hierarchically, in the case of molecular, with junction tree as
  the middle scaffolding.
