.. _tutorials3-index:

Generative models
--------------------

* **DGMG** `[paper] <https://arxiv.org/abs/1803.03324>`__ `[tutorial]
  <3_generative_model/5_dgmg.html>`__ `[PyTorch code]
  <https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgmg>`__:
  This model belongs to the family that deals with structural
  generation. Deep generative models of graphs (DGMG) uses a state-machine approach. 
  It is also very challenging because, unlike Tree-LSTM, every
  sample has a dynamic, probability-driven structure that is not available
  before training. You can progressively leverage intra- and
  inter-graph parallelism to steadily improve the performance.
