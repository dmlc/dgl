.. _guide-message-passing-api:

2.1 Built-in Functions and Message Passing APIs
-----------------------------------------------

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
the corresponding nodes and edges. The list of supported built-in functions
can be found in :ref:`api-built-in`. For example, to add the ``hu`` feature from src
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
:meth:`~dgl.DGLGraph.apply_edges`.
The parameters for ``apply_edges`` are a message function and valid
edge type as described in the API Doc (by default, all edges will be updated). For
example:

.. code::

    import dgl.function as fn
    graph.apply_edges(fn.u_add_v('el', 'er', 'e'))

the interface to call node-wise computation is
:meth:`~dgl.DGLGraph.update_all`.
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