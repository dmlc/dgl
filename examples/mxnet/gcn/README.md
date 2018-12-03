Graph Convolutional Networks (GCN)
============

Paper link: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)
Author's code repo: [https://github.com/tkipf/gcn](https://github.com/tkipf/gcn)

The folder contains three different implementations using DGL.

Results
-------
These results are based on single-run training to minimize the cross-entropy loss of the first 20 examples in each class. We can see clear improvements of graph convolution networks (GCNs) over multi-layer perceptron (MLP) baselines. There are also some slight modifications from the original paper:

* We used more (up to 10) layers to demonstrate monotonic improvements as more neighbor information is used. Using GCN with more layers improves accuracy but can also increase the computational complexity. The original paper recommends n-layers=2 to balance speed and accuracy.
* We used concatenation of hidden units to account for multi-hop skip-connections. The original implementation used simple additions (while the original paper omitted this detail). We feel concatenation is superior because all neighboring information is presented without additional modeling assumptions.
* After the concatenation, we used a recursive model such that the (k+1)-th layer, storing information up to the (k+1)-distant neighbor, depends on the concatenation of all 1-to-k layers. However, activation is only applied to the new information in the concatenations.

```
# Final accuracy 75.34% MLP without GCN
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 86.57% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 84.42% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python examples/mxnet/gcn/gcn_batch.py --dataset "citeseer" --n-epochs 200 --gpu 1 --n-layers 10
```

```
# Final accuracy 40.62% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 92.63% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 86.60% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "cora" --n-epochs 200 --gpu 1 --n-layers 10
```

```
# Final accuracy 72.97% MLP without GCN
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 0

# Final accuracy 88.33% with 10-layer GCN (symmetric normalization)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 10 --normalization 'sym' --self-loop

# Final accuracy 83.80% with 10-layer GCN (unnormalized)
DGLBACKEND=mxnet python3 examples/mxnet/gcn/gcn_batch.py --dataset "pubmed" --n-epochs 200 --gpu 1 --n-layers 10
```

Batched GCN (gcn_batch.py)
-----------

* The message function `gcn_msg` computes the message for a batch of edges. Here, the `src` argument is the batched representation of the source endpoints of the edges. The function simply returns the source node representations.
  ```python
  def gcn_msg(edge):
    # edge.src contains the node data on the source node.
    # It's a dictionary. edge.src['h'] is a tensor of shape (B, D).
    # B is the number of edges being batched.
    return {'m': edge.src['h']}
  ```
* The reduce function `gcn_reduce` also accumulates messages for a batch of nodes. We batch the messages on the second dimension fo the `msgs` argument:
  ```python
  def gcn_reduce(node):
    # node.mailbox contains messages from `gcn_msg`.
    # node.mailbox['m'] is a tensor of shape (B, deg, D). B is the number of nodes in the batch;
    #  deg is the number of messages; D is the message tensor dimension. DGL gaurantees
    #  that all the nodes in a batch have the same in-degrees (through "degree-bucketing").
    # Reduce on the second dimension is equal to sum up all the in-coming messages.
    return {'accum': mx.nd.sum(node.mailbox['m'], 1)}
  ```

* The update function `NodeUpdateModule` computes the new new node representation `h` using non-linear transformation on the reduced messages.
  ```python
  class NodeUpdateModule(gluon.Block):
    def __init__(self, out_feats, activation=None):
      super(NodeUpdateModule, self).__init__()
      self.linear = gluon.nn.Dense(out_feats, activation=activation)

    def forward(self, node):
      accum = self.linear(node.data['accum'])
      return {'h': mx.nd.concat(node.data['h'], accum, dim=1)}
  ```

After defining the functions on each node/edge, the message passing is triggered by calling `update_all` on the DGLGraph object (in GCN module).

```python
self.g.update_all(gcn_msg, gcn_reduce, layer)`
```
