.. _tutorials4-index:


Revisit classic models from a graph perspective
-------------------------------------------------------

* **Capsule** `[paper] <https://arxiv.org/abs/1710.09829>`__ `[tutorial]
  <4_old_wines/2_capsule.html>`__ `[PyTorch code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/capsule>`__:
  This new computer vision model has two key ideas. First, enhancing the feature
  representation in a vector form (instead of a scalar) called *capsule*. Second,
  replacing max-pooling with dynamic routing. The idea of dynamic routing is to
  integrate a lower level capsule to one or several higher level capsules
  with non-parametric message-passing. A tutorial shows how the latter can be 
  implemented with DGL APIs.


* **Transformer** `[paper] <https://arxiv.org/abs/1706.03762>`__ `[tutorial] <4_old_wines/7_transformer.html>`__ 
  `[PyTorch code] <https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer>`__ and **Universal Transformer** 
  `[paper] <https://arxiv.org/abs/1807.03819>`__ `[tutorial] <4_old_wines/7_transformer.html>`__
  `[PyTorch code] <https://github.com/dmlc/dgl/tree/master/examples/pytorch/transformer/modules/act.py>`__:
  These two models replace recurrent neural networks (RNNs) with several layers of multi-head attention to
  encode and discover structures among tokens of a sentence. These attention
  mechanisms are similarly formulated as graph operations with message-passing.
