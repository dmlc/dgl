.. currentmodule:: dgl

Built-in message passing functions
=================================

In DGL, message passing is expressed by two APIs:

- ``send(edges, message_func)`` for computing the messages along the given edges.
- ``recv(nodes, reduce_func)`` for collecting the incoming messages, perform aggregation and so on.

Although the two-stage abstraction can cover all the models that are defined in the message
passing paradigm, it is inefficient because it requires storing explicit messages. See the DGL 
`blog post <https://www.dgl.ai/blog/2019/05/04/kernel.html>`_ for more
details and performance results.

Our solution, also explained in the blog post, is to fuse the two stages into one kernel so no
explicit messages are generated and stored. To achieve this, we recommend using our built-in
message and reduce functions so that DGL can analyze and map them to fused dedicated kernels. Here
are some examples (in PyTorch syntax).

.. code:: python
   
   import dgl
   import dgl.function as fn
   import torch as th
   g = ... # create a DGLGraph
   g.ndata['h'] = th.randn((g.number_of_nodes(), 10)) # each node has feature size 10
   g.edata['w'] = th.randn((g.number_of_edges(), 1))  # each edge has feature size 1
   # collect features from source nodes and aggregate them in destination nodes
   g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_sum'))
   # multiply source node features with edge weights and aggregate them in destination nodes
   g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.max('m', 'h_max'))
   # compute edge embedding by multiplying source and destination node embeddings
   g.apply_edges(fn.u_mul_v('h', 'h', 'w_new'))

``fn.copy_u``, ``fn.u_mul_e``, ``fn.u_mul_v`` are built-in message functions, while ``fn.sum``
and ``fn.max`` are built-in reduce functions. We use ``u``, ``v`` and ``e`` to represent
source nodes, destination nodes, and edges among them, respectively. Hence, ``copy_u`` copies the source
node data as the messages, ``u_mul_e`` multiplies source node features with edge features, for example.

To define a unary message function (e.g. ``copy_u``) specify one input feature name and one output
message name. To define a binary message function (e.g. ``u_mul_e``) specify
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

Broadcasting is supported for binary message function, which means the tensor arguments
can be automatically expanded to be of equal sizes. The supported broadcasting semantic
is standard and matches `NumPy <https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html>`_
and `PyTorch <https://pytorch.org/docs/stable/notes/broadcasting.html>`_. If you are not familiar
with broadcasting, see the linked topics to learn more. In the
above example, ``fn.u_mul_e`` will perform broadcasted multiplication automatically because
the node feature ``'h'`` and the edge feature ``'w'`` are of different shapes, but they can be broadcast.

All DGL's built-in functions support both CPU and GPU and backward computation so they
can be used in any `autograd` system. Also, built-in functions can be used not only in ``update_all``
or ``apply_edges`` as shown in the example, but wherever message and reduce functions are
required (e.g. ``pull``, ``push``, ``send_and_recv``).

Here is a cheatsheet of all the DGL built-in functions.

+-------------------------+-----------------------------------------------------------------+-----------------------+
| Category                | Functions                                                       | Memo                  |
+=========================+=================================================================+=======================+
| Unary message function  | ``copy_u``                                                      |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``copy_e``                                                      |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``copy_src``                                                    |  alias of ``copy_u``  |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``copy_edge``                                                   |  alias of ``copy_e``  |
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
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``src_mul_edge``                                                |  alias of ``u_mul_e`` |
+-------------------------+-----------------------------------------------------------------+-----------------------+
| Reduce function         | ``max``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``min``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``sum``                                                         |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``prod``                                                        |                       |
|                         +-----------------------------------------------------------------+-----------------------+
|                         | ``mean``                                                        |                       |
+-------------------------+-----------------------------------------------------------------+-----------------------+

Next Step
---------
* To learn how built-in functions are used to implement Graph Neural
  Network layers See the :mod:`dgl.nn` module.
