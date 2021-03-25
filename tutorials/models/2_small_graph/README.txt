.. _tutorials2-index:

Batching many small graphs
-------------------------------

* **Tree-LSTM** `[paper] <https://arxiv.org/abs/1503.00075>`__ `[tutorial]
  <2_small_graph/3_tree-lstm.html>`__ `[PyTorch code]
  <https://github.com/dmlc/dgl/blob/master/examples/pytorch/tree_lstm>`__:
  Sentences have inherent structures that are thrown
  away by treating them simply as sequences. Tree-LSTM is a powerful model
  that learns the representation by using prior syntactic structures such as a parse-tree.
  The challenge in training is that simply by padding
  a sentence to the maximum length no longer works. Trees of different
  sentences have different sizes and topologies. DGL solves this problem by
  adding the trees to a bigger container graph, and then using message-passing
  to explore maximum parallelism. Batching is a key API for this.
