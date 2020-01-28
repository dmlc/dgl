## DGL Sampler

This directory contains the implementations for graph sampling routines in 0.5+.

### Code Hierarchy

#### Random walks:

* `randomwalks.h:`
  * `randomwalks_cpu.h:GenericRandomWalk(hg, seeds, max_num_steps, step)`
    * `metapath_randomwalk.h:RandomWalk(hg, seeds, metapath, prob, terminate)`
