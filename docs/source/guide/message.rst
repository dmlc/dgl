.. _guide-message-passing:

Chapter 2: Message Passing
================================

Message Passing Paradigm
------------------------

Let :math:`x_v\in\mathbb{R}^{d_1}` be the feature for node :math:`v`,
and :math:`w_{e}\in\mathbb{R}^{d_2}` be the feature for edge
:math:`({u}, {v})`. The **message passing paradigm** defines the
following node-wise and edge-wise computation at step :math:`t+1`:

.. math::  \text{Edge-wise: } m_{e}^{(t+1)} = \phi \left( x_v^{(t)}, x_u^{(t)}, w_{e}^{(t)} \right) , ({u}, {v},{e}) \in \mathcal{E}.

.. math::  \text{Node-wise: } x_v^{(t+1)} = \psi \left(x_v^{(t)}, \rho\left(\left\lbrace m_{e}^{(t+1)} : ({u}, {v},{e}) \in \mathcal{E} \right\rbrace \right) \right).

In the above equations, :math:`\phi` is a **message function**
defined on each edge to generate a message by combining the edge feature
with the features of its incident nodes; :math:`\psi` is an
**update function** defined on each node to update the node feature
by aggregating its incoming messages using the **reduce function**
:math:`\rho`.

Built-in Functions and Message Passing APIs
-------------------------------------------

In DGL, **message function** takes a single argument ``edges``,
which has three members ``src``, ``dst`` and ``data``, to access
features of source node, destination node, and edge, respectively.

**reduce function** takes a single argument ``nodes``. A node can
access its ``mailbox`` to collect the messages its neighbors send to it
through edges. Some of the most common reduce operations include ``sum``,
``max``, ``min``, etc.

**update function** takes a single argument ``nodes``. This function
operates on the aggregation result from ``reduce function``, typically
combined with a node’s feature at the the last step, and save the output
as a node feature.

DGL has implemented commonly used message functions and reduce functions
as **built-in** in the namespace ``dgl.function``. In general, we
suggest using built-in functions **whenever possible** since they are
heavily optimized and automatically handle dimension broadcasting.

If your message passing functions cannot be implemented with built-ins,
you can implement user-defined message/reduce function (aka. **UDF**).

Built-in message functions can be unary or binary. We support ``copy``
for unary for now. For binary funcs, we now support ``add``, ``sub``,
``mul``, ``div``, ``dot``. The naming convention for message
built-in funcs is ``u`` represents ``src`` nodes, ``v`` represents
``dst`` nodes, ``e`` represents ``edges``. The parameters for those
functions are strings indicating the input and output field names for
the corresponding nodes and edges. Here is the
`list <https://docs.dgl.ai/api/python/function.html#>`__ of supported
built-in functions. For example, to add the ``hu`` feature from src
nodes and ``hv`` feature from dst nodes then save the result on the edge
at ``he`` field, we can use built-in function
``dgl.function.u_add_v('hu', 'hv', 'he')`` this is equivalent to the
Message UDF:

.. code::

    def message_func(edges):
         return {'he': edges.src['hu'] + edges.dst['hv']}

Built-in reduce functions support operations ``sum``, ``max``, ``min``,
``prod`` and ``mean``. Reduce functions usually have two parameters, one
for field name in ``mailbox``, one for field name in destination, both
are strings. For example, ``dgl.function.sum('m', 'h')`` is equivalent
to the Reduce UDF that sums up the message ``m``:

.. code::

    import torch
    def reduce_func(nodes):
         return {'h': torch.sum(nodes.mailbox['m'], dim=1)}

In DGL, the interface to call edge-wise computation is
`apply_edges() <https://docs.dgl.ai/generated/dgl.DGLGraph.apply_edges.html>`__.
The parameters for ``apply_edges`` are a message function and valid
edge type (see
`send() <https://docs.dgl.ai/en/0.4.x/generated/dgl.DGLGraph.send.html#dgl.DGLGraph.send>`_
for valid edge types, by default, all edges will be updated). For
example:

.. code::

    import dgl.function as fn
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

the interface to call node-wise computation is
`update_all() <https://docs.dgl.ai/generated/dgl.DGLGraph.update_all.html>`__.
The parameters for ``update_all`` are a message function, a
reduce function and a update function. update function can
be called outside of ``update_all`` by leaving the third parameter as
empty. This is suggested since the update function can usually be
written as pure tensor operations to make the code concise. For
example：

.. code::

    def updata_all_example(graph):
        # store the result in graph.ndata['ft']
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                        fn.sum('m', 'ft'))
        # Call update function outside of update_all
        final_ft = graph.ndata['ft'] * 2
        return final_ft

This call will generate the message ``m`` by multiply src node feature
``ft`` and edge feature ``a``, sum up the message ``m`` to update node
feature ``ft``, finally multiply ``ft`` by 2 to get the result
``final_ft``. After the call, the intermediate message ``m`` will be
cleaned. The math formula for the above function is:

.. math::  {final\_ft}_i = 2 * \sum_{j\in\mathcal{N}(i)} ({ft}_j * a_{ij})

``update_all`` is a high-level API that merges message generation,
message reduction and node update in a single call, which leaves room
for optimizations, as explained below.

Writing Efficient Message Passing Codes
----------------------------------------------

DGL optimized memory consumption and computing speed for message
passing. The optimization includes:

-  Merge multiple kernels in a single one: This is achieved by using
   ``update_all`` to call multiple built-in functions at once.
   (Speed optimization)

-  Parallelism on nodes and edges: DGL abstracts edge-wise computation
   ``apply_edges`` as a generalized sampled dense-dense matrix
   multiplication (**gSDDMM**) operation and parallelize the computing
   across edges. Likewise, DGL abstracts node-wise computation
   ``update_all`` as a generalized sparse-dense matrix multiplication
   (**gSPMM**) operation and parallelize the computing across nodes.
   (Speed optimization)

-  Avoid unnecessary memory copy into edges: To generate a message that
   requires the feature from source and destination node, one option is
   to copy the source and destination node feature into that edge. For
   some graphs, the number of edges is much larger than the number of
   nodes. This copy can be costly. DGL built-in message functions
   avoid this memory copy by sampling out the node feature using entry
   index. (Memory and speed optimization)

-  Avoid materializing feature vectors on edges: the complete message
   passing process includes message generation, message reduction and
   node update. In ``update_all`` call, message function and reduce
   function are merged into one kernel if those functions are
   built-in. There is no message materialization on edges. (Memory
   optimization)

According to the above, a common practise to leverage those
optimizations is to construct your own message passing functionality as
a combination of ``update_all`` calls with built-in functions as
parameters.

For some cases like
`GAT <https://github.com/dmlc/dgl/blob/master/python/dgl/nn/pytorch/conv/gatconv.py>`__
where we have to save message on the edges, we need to call
``apply_edges`` with built-in functions. Sometimes the message on
the edges can be high dimensional, which is memory consuming. We suggest
keeping the edata dimension as low as possible.

Here’s an example on how to achieve this by spliting operations on the
edges to nodes. The option does the following: concatenate the ``src``
feature and ``dst`` feature, then apply a linear layer, i.e.
:math:`W\times (u || v)`. The ``src`` and ``dst`` feature dimension is
high, while the linear layer output dimension is low. A straight forward
implementation would be like:

.. code::

    linear = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim*2)))
    def concat_message_function(edges):
        {'cat_feat': torch.cat([edges.src.ndata['feat'], edges.dst.ndata['feat']])}
    g.apply_edges(concat_message_function)
    g.edata['out'] = g.edata['cat_feat'] * linear

The suggested implementation will split the linear operation into two,
one applies on ``src`` feature, the other applies on ``dst`` feature.
Add the output of the linear operations on the edges at the final stage,
i.e. perform :math:`W_l\times u + W_r \times v`, since
:math:`W \times (u||v) = W_l \times u + W_r \times v`, where :math:`W_l`
and :math:`W_r` are the left and the right half of the matrix :math:`W`,
respectively:

.. code::

    linear_src = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim)))
    linear_dst = nn.Parameter(th.FloatTensor(size=(1, node_feat_dim)))
    out_src = g.ndata['feat'] * linear_src
    out_dst = g.ndata['feat'] * linear_dst
    g.srcdata.update({'out_src': out_src})
    g.dstdata.update({'out_dst': out_dst})
    g.apply_edges(fn.u_add_v('out_src', 'out_dst', 'out'))

The above two implementations are mathematically equivalent. The later
one is much efficient because we do not need to save feat_src and
feat_dst on edges, which is not memory-efficient. Plus, addition could
be optimized with DGL’s built-in function ``u_add_v``, which further
speeds up computation and saves memory footprint.

Apply Message Passing On Part Of The Graph
-----------------------------------------------

If we only want to update part of the nodes in the graph, the practice
is to create a subgraph by providing the ids for the nodes we want to
include in the update, then call ``update_all`` on the subgraph. For
example:

.. code::

    nid = [0, 2, 3, 6, 7, 9]
    sg = g.subgraph(nid)
    sg.update_all(message_func, reduce_func, apply_node_func)

This is a common usage in mini-batch training. Check `mini-batch
training <https://docs.dgl.ai/generated/guide/minibatch.html>`__ user guide for more detailed
usages.

Apply Edge Weight In Message Passing
----------------------------------------

A commonly seen practice in GNN modeling is to apply edge weight on the
message before message aggregation, for examples, in
`GAT <https://arxiv.org/pdf/1710.10903.pdf>`__ and some `GCN
variants <https://arxiv.org/abs/2004.00445>`__. In DGL, the way to
handle this is:

-  Save the weight as edge feature.
-  Multiply the edge feature with src node feature in message function.

For example:

.. code::

    graph.edata['a'] = affinity
    graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                     fn.sum('m', 'ft'))

In the above, we use affinity as the edge weight. The edge weight should
usually be a scalar.

Message Passing on Heterogeneuous Graph
---------------------------------------

`Heterogeneous
graphs <https://docs.dgl.ai/tutorials/basics/5_hetero.html>`__, or
heterographs for short, are graphs that contain different types of nodes
and edges. The different types of nodes and edges tend to have different
types of attributes that are designed to capture the characteristics of
each node and edge type. Within the context of graph neural networks,
depending on their complexity, certain node and edge types might need to
be modeled with representations that have a different number of
dimensions.

The message passing on heterographs can be split into two parts:

1. Message computation and aggregation within each relation r.
2. Reduction that merges the results on the same node type from multiple
   relationships.

DGL’s interface to call message passing on heterographs is
:meth:`~dgl.DGLGraph.multi_update_all`.
``multi_update_all`` takes a dictionary containing the parameters for
``update_all`` within each relation using relation as the key, and a
string represents the cross type reducer. The reducer can be one of
``sum``, ``min``, ``max``, ``mean``, ``stack``. Here’s an example:

.. code::

    for c_etype in G.canonical_etypes:
        srctype, etype, dsttype = c_etype
        Wh = self.weight[etype](feat_dict[srctype])
        # Save it in graph for message passing
        G.nodes[srctype].data['Wh_%s' % etype] = Wh
        # Specify per-relation message passing functions: (message_func, reduce_func).
        # Note that the results are saved to the same destination feature 'h', which
        # hints the type wise reducer for aggregation.
        funcs[etype] = (fn.copy_u('Wh_%s' % etype, 'm'), fn.mean('m', 'h'))
    # Trigger message passing of multiple types.
    G.multi_update_all(funcs, 'sum')
    # return the updated node feature dictionary
    return {ntype : G.nodes[ntype].data['h'] for ntype in G.ntypes}
