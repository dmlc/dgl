.. _apifunction:

.. currentmodule:: dgl.function

dgl.function
==================================

This subpackage hosts all the **built-in functions** provided by DGL. Built-in functions
are DGL's recommended way to express different types of :ref:`guide-message-passing` computation
(i.e., via :func:`~dgl.DGLGraph.update_all`) or computing edge-wise features from
node-wise features (i.e., via :func:`~dgl.DGLGraph.apply_edges`). Built-in functions
describe the node-wise and edge-wise computation in a symbolic way without any
actual computation, so DGL can analyze and map them to efficient low-level kernels.
Here are some examples:

.. code:: python

   import dgl
   import dgl.function as fn
   import torch as th
   g = ... # create a DGLGraph
   g.ndata['h'] = th.randn((g.num_nodes(), 10)) # each node has feature size 10
   g.edata['w'] = th.randn((g.num_edges(), 1))  # each edge has feature size 1
   # collect features from source nodes and aggregate them in destination nodes
   g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
   # multiply source node features with edge weights and aggregate them in destination nodes
   g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'h_max'))
   # compute edge embedding by multiplying source and destination node embeddings
   g.apply_edges(fn.u_mul_v('h', 'h', 'w_new'))

``fn.copy_u``, ``fn.u_mul_e``, ``fn.u_mul_v`` are built-in message functions, while ``fn.sum``
and ``fn.max`` are built-in reduce functions. DGL's convention is to use ``u``, ``v``
and ``e`` to represent source nodes, destination nodes, and edges, respectively.
For example, ``copy_u`` tells DGL to copy the source node data as the messages;
``u_mul_e`` tells DGL to multiply source node features with edge features.

To define a unary message function (e.g. ``copy_u``), specify one input feature name and one output
message name. To define a binary message function (e.g. ``u_mul_e``), specify
two input feature names and one output message name. During the computation,
the message function will read the data under the given names, perform computation, and return
the output using the output name. For example, the above ``fn.u_mul_e('h', 'w', 'm')`` is
the same as the following user-defined function:

.. code:: python

   def udf_u_mul_e(edges):
      return {'m' : edges.src['h'] * edges.data['w']}

To define a reduce function, one input message name and one output node feature name
need to be specified. For example, the above ``fn.max('m', 'h_max')`` is the same as the
following user-defined function:

.. code:: python

   def udf_max(nodes):
      return {'h_max' : th.max(nodes.mailbox['m'], 1)[0]}

All binary message function supports **broadcasting**, a mechanism for extending element-wise
operations to tensor inputs with different shapes. DGL generally follows the standard
broadcasting semantic by `NumPy <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
and `PyTorch <https://pytorch.org/docs/stable/notes/broadcasting.html>`_. Below are some
examples:

.. code:: python

   import dgl
   import dgl.function as fn
   import torch as th
   g = ... # create a DGLGraph

   # case 1
   g.ndata['h'] = th.randn((g.num_nodes(), 10))
   g.edata['w'] = th.randn((g.num_edges(), 1))
   # OK, valid broadcasting between feature shapes (10,) and (1,)
   g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
   g.ndata['h_new']  # shape: (g.num_nodes(), 10)

   # case 2
   g.ndata['h'] = th.randn((g.num_nodes(), 5, 10))
   g.edata['w'] = th.randn((g.num_edges(), 10))
   # OK, valid broadcasting between feature shapes (5, 10) and (10,)
   g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))
   g.ndata['h_new']  # shape: (g.num_nodes(), 5, 10)

   # case 3
   g.ndata['h'] = th.randn((g.num_nodes(), 5, 10))
   g.edata['w'] = th.randn((g.num_edges(), 5))
   # NOT OK, invalid broadcasting between feature shapes (5, 10) and (5,)
   # shapes are aligned from right
   g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'h_new'))

   # case 3
   g.ndata['h1'] = th.randn((g.num_nodes(), 1, 10))
   g.ndata['h2'] = th.randn((g.num_nodes(), 5, 1))
   # OK, valid broadcasting between feature shapes (1, 10) and (5, 1)
   g.apply_edges(fn.u_add_v('h1', 'h2', 'x'))  # apply_edges also supports broadcasting
   g.edata['x']  # shape: (g.num_edges(), 5, 10)

   # case 4
   g.ndata['h1'] = th.randn((g.num_nodes(), 1, 10, 128))
   g.ndata['h2'] = th.randn((g.num_nodes(), 5, 1, 128))
   # OK, u_dot_v supports broadcasting but requires the last dimension to match
   g.apply_edges(fn.u_dot_v('h1', 'h2', 'x'))
   g.edata['x']  # shape: (g.num_edges(), 5, 10, 1)


.. _api-built-in:

DGL Built-in Function
-------------------------

Here is a cheatsheet of all the DGL built-in functions.

+-------------------------+-----------------------------------------------------------------+-----------------------+
| Category                | Functions                                                       | Memo                  |
+=========================+=================================================================+=======================+
| Unary message function  | ``copy_u``                                                      |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``copy_e``                                                      |                       |
+-------------------------+-----------------------------------------------------------------+-----------------------+
| Binary message function | ``u_add_v``, ``u_sub_v``, ``u_mul_v``, ``u_div_v``, ``u_dot_v`` |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``u_add_e``, ``u_sub_e``, ``u_mul_e``, ``u_div_e``, ``u_dot_e`` |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``v_add_u``, ``v_sub_u``, ``v_mul_u``, ``v_div_u``, ``v_dot_u`` |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``v_add_e``, ``v_sub_e``, ``v_mul_e``, ``v_div_e``, ``v_dot_e`` |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``e_add_u``, ``e_sub_u``, ``e_mul_u``, ``e_div_u``, ``e_dot_u`` |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``e_add_v``, ``e_sub_v``, ``e_mul_v``, ``e_div_v``, ``e_dot_v`` |                       |
+-------------------------+-----------------------------------------------------------------+-----------------------+
| Reduce function         | ``max``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``min``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``sum``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``mean``                                                        |                       |
+-------------------------+-----------------------------------------------------------------+-----------------------+

Message functions
-----------------

.. autosummary::
    :toctree: ../../generated/

    copy_u
    copy_e
    u_add_v
    u_sub_v
    u_mul_v
    u_div_v
    u_add_e
    u_sub_e
    u_mul_e
    u_div_e
    v_add_u
    v_sub_u
    v_mul_u
    v_div_u
    v_add_e
    v_sub_e
    v_mul_e
    v_div_e
    e_add_u
    e_sub_u
    e_mul_u
    e_div_u
    e_add_v
    e_sub_v
    e_mul_v
    e_div_v
    u_dot_v
    u_dot_e
    v_dot_e
    v_dot_u
    e_dot_u
    e_dot_v

Reduce functions
----------------

.. autosummary::
    :toctree: ../../generated/

    sum
    max
    min
    mean
