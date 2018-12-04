.. _tutorials1-index:

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