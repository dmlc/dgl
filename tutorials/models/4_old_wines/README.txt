.. _tutorials4-index:


Old (new) wines in new bottle
=============================

* **Capsule** `[paper] <https://arxiv.org/abs/1710.09829>`__ `[tutorial]
  <4_old_wines/2_capsule.html>`__ `[code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/capsule>`__:
  this new computer vision model has two key ideas -- enhancing the feature
  representation in a vector form (instead of a scalar) called *capsule*, and
  replacing max-pooling with dynamic routing. The idea of dynamic routing is to
  integrate a lower level capsule to one (or several) of a higher level one
  with non-parametric message-passing. We show how the later can be nicely
  implemented with DGL APIs.


* **Transformer** `[paper] <https://arxiv.org/abs/1706.03762>`__ `[tutorial] <4_old_wines/7_transformer.html>`__ 
  `[code] <https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer>`__ and **Universal Transformer** 
  `[paper] <https://arxiv.org/abs/1807.03819>`__ `[tutorial] <4_old_wines/7_transformer.html>`__
  `[code] <https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer/modules/act.py>`__:
  these two models replace RNN with several layers of multi-head attention to
  encode and discover structures among tokens of a sentence. These attention
  mechanisms can similarly formulated as graph operations with
  message-passing.
