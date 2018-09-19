Graph Convolutional Networks (GCN)
============

Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

The folder contains three different implementations using DGL.

Naive GCN (gcn.py)
-------
The model is defined in the finest granularity (aka on *one* edge and *one* node).

* The message function `gcn_msg` computes the message for one edge. It simply returns the `h` representation of the source node.
  ```python
  def gcn_msg(src, edge):
    # src['h'] is a tensor of shape (D,). D is the feature length.
    return src['h']
  ```
* The reduce function `gcn_reduce` accumulates the incoming messages for one node. The `msgs` argument is a list of all the messages. In GCN, the incoming messages are summed up.
  ```python
  def gcn_reduce(node, msgs):
    # msgs is a list of in-coming messages.
    return sum(msgs)
  ```
* The update function `NodeUpdateModule` computes the new new node representation `h` using non-linear transformation on the reduced messages.
  ```python
  class NodeUpdateModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
      super(NodeUpdateModule, self).__init__()
      self.linear = nn.Linear(in_feats, out_feats)
      self.activation = activation

    def forward(self, node, accum):
      # accum is a tensor of shape (D,).
      h = self.linear(accum)
      if self.activation:
          h = self.activation(h)
      return {'h' : h}
  ```

After defining the functions on each node/edge, the message passing is triggered by calling `update_all` on the DGLGraph object (in GCN module).

Batched GCN (gcn_batch.py)
-----------
Defining the model on only one node and edge makes it hard to fully utilize GPUs. As a result, we allow users to define model on a *batch of* nodes and edges.

* The message function `gcn_msg` computes the message for a batch of edges. Here, the `src` argument is the batched representation of the source endpoints of the edges. The function simply returns the source node representations.
  ```python
  def gcn_msg(src, edge):
    # src is a tensor of shape (B, D). B is the number of edges being batched.
    return src
  ```
* The reduce function `gcn_reduce` also accumulates messages for a batch of nodes. We batch the messages on the second dimension fo the `msgs` argument:
  ```python
  def gcn_reduce(node, msgs):
    # The msgs is a tensor of shape (B, deg, D). B is the number of nodes in the batch;
    #  deg is the number of messages; D is the message tensor dimension. DGL gaurantees
    #  that all the nodes in a batch have the same in-degrees (through "degree-bucketing").
    # Reduce on the second dimension is equal to sum up all the in-coming messages.
    return torch.sum(msgs, 1)
  ```
* The update module is similar. The first dimension of each tensor is the batch dimension. Since PyTorch operation is usually aware of the batch dimension, the code is the same as the naive GCN.

Triggering message passing is also similar. User needs to set `batchable=True` to indicate that the functions all support batching.
```python
self.g.update_all(gcn_msg, gcn_reduce, layer, batchable=True)`
```

Batched GCN with spMV optimization (gcn_spmv.py)
-----------
Batched computation is much more efficient than naive vertex-centric approach, but is still not ideal. For example, the batched message function needs to look up source node data and save it on edges. Such kind of lookups is very common and incurs extra memory copy operations. In fact, the message and reduce phase of GCN model can be fused into one sparse-matrix-vector multiplication (spMV). Therefore, DGL provides many built-in message/reduce functions so we can figure out the chance of optimization. In gcn_spmv.py, user only needs to write update module and trigger the message passing as follows:
```python
self.g.update_all('from_src', 'sum', layer, batchable=True)
```
Here, `'from_src'` and `'sum'` are the builtin message and reduce function.
