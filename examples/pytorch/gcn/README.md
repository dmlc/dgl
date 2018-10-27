Graph Convolutional Networks (GCN)
============

- Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
- Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn). Note that the original code is 
implemented with Tensorflow for the paper. 

The folder contains two different implementations using DGL.

Batched GCN (gcn.py)
-----------
Defining the model on only one node and edge makes it hard to fully utilize GPUs. As a result, we allow users to define model on a *batch of* nodes and edges.

* The message function `gcn_msg` computes the message for a batch of edges. Here, the `src` argument is the batched representation of the source endpoints of the edges. The function simply returns the source node representations.
  ```python
  def gcn_msg(src, edge):
    # src is a tensor of shape (B, D). B is the number of edges being batched.
    return {'m' : src['h']}
  ```
* The reduce function `gcn_reduce` also accumulates messages for a batch of nodes. We batch the messages on the second dimension for the `msgs` argument, 
which for example can correspond to the neighbors of the nodes:
  ```python
  def gcn_reduce(node, msgs):
    # The msgs is a tensor of shape (B, deg, D). B is the number of nodes in the batch;
    #  deg is the number of messages; D is the message tensor dimension. DGL gaurantees
    #  that all the nodes in a batch have the same in-degrees (through "degree-bucketing").
    # Reduce on the second dimension is equal to sum up all the in-coming messages.
    return {'h' : torch.sum(msgs['m'], 1)}
  ```
* The update module is similar. The first dimension of each tensor is the batch dimension. Since PyTorch operation is usually aware of the batch dimension, the code is the same as the naive GCN.

Triggering message passing is also similar. 
```python
self.g.update_all(gcn_msg, gcn_reduce, layer)`
```

Batched GCN with spMV optimization (gcn_spmv.py)
-----------
Batched computation is much more efficient than naive vertex-centric approach, but is still not ideal. For example, the batched message function needs to look up source node data and save it on edges. Such kind of lookups is very common and incurs extra memory copy operations. In fact, the message and reduce phase of GCN model can be fused into one sparse-matrix-vector multiplication (spMV). Therefore, DGL provides many built-in message/reduce functions so we can figure out the chance of optimization. In gcn_spmv.py, user only needs to write update module and trigger the message passing as follows:
```python
self.g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'), layer)
```
Here, `'fn.copy_src'` and `'fn.sum'` are the builtin message and reduce functions that perform the same operations as `gcn_msg` and `gcn_reduce` in gcn.py.
