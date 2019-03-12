DGL release and change logs
==========

Refer to the roadmap issue for the on-going versions and features.

0.2
---
Major release that includes many features, bugfix and performance improvement.
Speed of GCN model on Pubmed dataset has been improved by **4.19x**! Speed of
RGCN model on Mutag dataset has been improved by **7.35x**! Important new
feature: **graph sampling APIs**.

Update details:

# Model examples
- [x] TreeLSTM w/ MXNet (PR #279 by @szha )
- [x] GraphSage (@ZiyueHuang )
- [x] Improve GAT model speed (PR #348 by @jermainewang )

# Core system improvement
- [x] Immutable CSR graph structure (PR #342 by @zheng-da )
  - [x] Finish remaining functionality (Issue #369, PR #404 by @yzh119)
- [x] Nodeflow data structure (PR #361 by @zheng-da )
- [x] Neighbor sampler (PR #322 )
- [x] Layer-wise sampler (PR #362 by @GaiYu0 )
- [x] Multi-GPU support by data parallelism (PR #356 #338 by @ylfdq1118 )
- [x] More dataset:
  - [x] Reddit dataset loader (PR #372 by @ZiyueHuang )
  - [x] PPI dataset loader (PR #395 by @sufeidechabei )
  - [x] Mini graph classification dataset (PR #364 by @mufeili )
- [x] NN modules (PR #406 by @jermainewang @mufeili)
  - [x] GraphConv layer
  - [x] Edge softmax layer
- [x] Edge group apply API (PR #358 by @VoVAllen )
- [x] Reversed graph and transform.py module (PR #331 by @mufeili )
- [x] Max readout (PR #341 by @mufeili )
- [x] Random walk APIs (PR #392 by @BarclayII )

# Tutorial/Blog
- [x] Batched graph classification in DGL (PR #360 by @mufeili )
- [x] Understanding GAT (@sufeidechabei )

# Project improvement
- [x] Python lint check (PR #330 by @jermainewang )
- [x] Win CI (PR #324 by @BarclayII )
- [x] Auto doc build (by @VoVAllen )
- [x] Unify tests for different backends (PR #333 by @BarclayII )

0.1.3
-----
Bug fix
* Compatible with Pytorch v1.0
* Bug fix in networkx graph conversion.

0.1.2
-----
First open release.
* Basic graph APIs.
* Basic message passing APIs.
* Pytorch backend.
* MXNet backend.
* Optimization using SPMV.
* Model examples w/ Pytorch:
  - GCN
  - GAT
  - JTNN
  - DGMG
  - Capsule
  - LGNN
  - RGCN
  - Transformer
  - TreeLSTM
* Model examples w/ MXNet:
  - GCN
  - GAT
  - RGCN
  - SSE
