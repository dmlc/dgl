Graph-based Neural Network Models
=================================

We developed DGL with a broad range of applications in mind. Building
state-of-art models forces us to think hard on the most common and useful APIs,
learn the hard lessons, and push the system design.

We have prototyped altogether 10 different models, all of them are ready to run
out-of-box and some of them are very new graph-based algorithms. In most of the
cases, they demonstrate the performance, flexibility, and expressiveness of
DGL. For where we still fall in short, these exercises point to future
directions.

We categorize the models below, providing links to the original code and
tutorial when appropriate. As will become apparent, these models stress the use
of different DGL APIs.

Graph Neural Network and its variant
------------------------------------

* **GCN** `[paper] <https://arxiv.org/abs/1609.02907>`__ `[tutorial] <models/1_gcn.html>`__
  `[code] <https://github.com/jermainewang/dgl/blob/master/examples/pytorch/gcn/gcn.py>`__:
  this is the vanilla GCN. The tutorial covers the basic uses of DGL APIs.

* **GAT** `[paper] <https://arxiv.org/abs/1710.10903>`__
  `[code] <https://github.com/jermainewang/dgl/blob/master/examples/pytorch/gat/gat.py>`__:
  the key extension of GAT w.r.t vanilla GCN is deploying multi-head attention
  among neighborhood of a node, thus greatly enhances the capacity and
  expressiveness of the model.

* **R-GCN** `[paper] <https://arxiv.org/abs/1703.06103>`__ `[tutorial] <models/4_rgcn.html>`__
  [code (wip)]: the key
  difference of RGNN is to allow multi-edges among two entities of a graph, and
  edges with distinct relationships are encoded differently. This is an
  interesting extension of GCN that can have a lot of applications of its own.

* **LGNN** `[paper] <https://arxiv.org/abs/1705.08415>`__ `[tutorial (wip)]` `[code (wip)]`:
  this model focuses on community detection by inspecting graph structures. It
  uses representations of both the orignal graph and its line-graph companion. In
  addition to demonstrate how an algorithm can harness multiple graphs, our
  implementation shows how one can judiciously mix vanilla tensor operation,
  sparse-matrix tensor operations, along with message-passing with DGL. 

* **SSE** `[paper] <http://proceedings.mlr.press/v80/dai18a/dai18a.pdf>`__ `[tutorial (wip)]`
  `[code] <https://github.com/jermainewang/dgl/blob/master/examples/mxnet/sse/sse_batch.py>`__:
  the emphasize here is *giant* graph that cannot fit comfortably on one GPU
  card. SSE is an example to illustrate the co-design of both algrithm and
  system: sampling to guarantee asymptotic covergence while lowering the
  complexity, and batching across samples for maximum parallelism.

Dealing with many small graphs
------------------------------

* **Tree-LSTM** `[paper] <https://arxiv.org/abs/1503.00075>`__ `[tutorial] <models/3_tree-lstm.html>`__
  `[code] <https://github.com/jermainewang/dgl/blob/master/examples/pytorch/tree_lstm/tree_lstm.py>`__:
  sentences of natural languages have inherent structures, which are thrown away
  by treating them simply as sequences. Tree-LSTM is a powerful model that learns
  the representation by leveraging prior syntactic structures (e.g. parse-tree).
  The challenge to train it well is that simply by padding a sentence to the
  maximum length no longer works, since trees of different sentences have
  different sizes and topologies. DGL solves this problem by throwing the trees
  into a bigger "container" graph, and use message-passing to explore maximum
  parallelism. The key API we use is batching.

Generative models
------------------------------

* **DGMG** `[paper] <https://arxiv.org/abs/1803.03324>`__ `[tutorial] <models/5_dgmg.html>`__
  `[code] <https://github.com/jermainewang/dgl/tree/master/examples/pytorch/dgmg>`__:
  this model belongs to the important family that deals with structural
  generation. DGMG is interesting because its state-machine approach is the most
  general. It is also very challenging because, unlike Tree-LSTM, every sample
  has a dynamic, probability-driven structure that is not available before
  training. We are able to progressively leverage intra- and inter-graph
  parallelism to steadily improve the performance.

* **JTNN** `[paper] <https://arxiv.org/abs/1802.04364>`__ `[code (wip)]`: unlike DGMG, this
  paper generates molecular graphs using the framework of variational
  auto-encoder. Perhaps more interesting is its approach to build structure
  hierarchically, in the case of molecular, with junction tree as the middle
  scaffolding.

Old (new) wines in new bottle
-----------------------------
* **Capsule** `[paper] <https://arxiv.org/abs/1710.09829>`__ `[tutorial] <models/2_capsule.html>`__
  `[code] <https://github.com/jermainewang/dgl/tree/master/examples/pytorch/capsule>`__: this new
  computer vision model has two key ideas -- enhancing the feature representation
  in a vector form (instead of a scalar) called *capsule*, and replacing
  maxpooling with dynamic routing. The idea of dynamic routing is to integrate a
  lower level capsule to one (or several) of a higher level one with
  non-parametric message-passing. We show how the later can be nicely implemented
  with DGL APIs.

* **Transformer** `[paper] <https://arxiv.org/abs/1706.03762>`__ `[tutorial (wip)]` `[code (wip)]` and
  **Universal Transformer** `[paper] <https://arxiv.org/abs/1807.03819>`__ `[tutorial (wip)]`
  `[code (wip)]`: these
  two models replace RNN with several layers of multi-head attention to encode
  and discover structures among tokens of a sentence. These attention mechanisms
  can similarly formulated as graph operations with message-passing. 
