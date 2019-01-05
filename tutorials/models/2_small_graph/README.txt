.. _tutorials2-index:

Dealing with many small graphs
==============================

* **Tree-LSTM** `[paper] <https://arxiv.org/abs/1503.00075>`__ `[tutorial]
  <2_small_graph/3_tree-lstm.html>`__ `[code]
  <https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm>`__:
  sentences of natural languages have inherent structures, which are thrown
  away by treating them simply as sequences. Tree-LSTM is a powerful model
  that learns the representation by leveraging prior syntactic structures
  (e.g. parse-tree). The challenge to train it well is that simply by padding
  a sentence to the maximum length no longer works, since trees of different
  sentences have different sizes and topologies. DGL solves this problem by
  throwing the trees into a bigger "container" graph, and use message-passing
  to explore maximum parallelism. The key API we use is batching.
