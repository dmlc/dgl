.. _apibackend:

.. currentmodule:: dgl.ops

dgl.ops
==================================

Frame-agnostic operators for message passing on graphs.

GSpMM functions
---------------

Generalized Sparse-Matrix Dense-Matrix Multiplication functions.
It *fuses* two steps into one kernel.

1. Computes messages by add/sub/mul/div source node and edge features,
   or copy node features to edges.
2. Aggregate the messages by sum/max/min/mean as the features on destination nodes.

Our implementation supports tensors on CPU/GPU in PyTorch/MXNet/Tensorflow
as input. All operators are equipped with autograd (computing the input gradients
given output gradient) and broadcasting (if the feature shape of operands do not
match, we first broadcast them to the same shape, then applies the binary
operators). Our broadcast semantics follows NumPy, please see
https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
for more details.

What do we mean by *fuses* is that the messages are not materialized on edges,
instead we compute the result on destination nodes directly, thus saving memory
cost. The space complexity of GSpMM operators is :math:`O(|N|D)` where :math:`|N|`
refers to the number of nodes in the graph, and :math:`D` refers to the feature
size (:math:`D=\prod_{i=1}^{N}D_i` if your feature is a multi-dimensional tensor).

The following is an example showing how GSpMM works (we use PyTorch as the backend
here, you can enjoy the same convenience on other frameworks by similar usage):

   >>> import dgl
   >>> import torch as th
   >>> import dgl.ops as F
   >>> g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))  # 3 nodes, 6 edges
   >>> x = th.ones(3, 2, requires_grad=True)
   >>> x
   tensor([[1., 1.],
           [1., 1.],
           [1., 1.]], requires_grad=True)
   >>> y = th.arange(1, 13).float().view(6, 2).requires_grad_()
   tensor([[ 1.,  2.],
           [ 3.,  4.],
           [ 5.,  6.],
           [ 7.,  8.],
           [ 9., 10.],
           [11., 12.]], requires_grad=True)
   >>> out_1 = F.u_mul_e_sum(g, x, y)
   >>> out_1  # (10, 12) = ((1, 1) * (3, 4)) + ((1, 1) * (7, 8))
   tensor([[ 1.,  2.],
           [10., 12.],
           [25., 28.]], grad_fn=<GSpMMBackward>)
   >>> out_1.sum().backward()
   >>> x.grad
   tensor([[12., 15.],
           [18., 20.],
           [12., 13.]])
   >>> y.grad
   tensor([[1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.],
           [1., 1.]])
   >>> out_2 = F.copy_u_sum(g, x)
   >>> out_2
   tensor([[1., 1.],
           [2., 2.],
           [3., 3.]], grad_fn=<GSpMMBackward>)
   >>> out_3 = F.u_add_e_max(g, x, y)
   >>> out_3
   tensor([[ 2.,  3.],
           [ 8.,  9.],
           [12., 13.]], grad_fn=<GSpMMBackward>)
   >>> y1 = th.rand(6, 4, 2, requires_grad=True)  # test broadcast
   >>> F.u_mul_e_sum(g, x, y1).shape  # (2,), (4, 2) -> (4, 2)
   torch.Size([3, 4, 2])

For all operators, the input graph could either be a homogeneous or a bipartite
graph.

.. autosummary::
    :toctree: ../../generated/

    gspmm
    u_add_e_sum
    u_sub_e_sum
    u_mul_e_sum
    u_div_e_sum
    u_add_e_max
    u_sub_e_max
    u_mul_e_max
    u_div_e_max
    u_add_e_min
    u_sub_e_min
    u_mul_e_min
    u_div_e_min
    u_add_e_mean
    u_sub_e_mean
    u_mul_e_mean
    u_div_e_mean
    copy_u_sum
    copy_e_sum
    copy_u_max
    copy_e_max
    copy_u_min
    copy_e_min 
    copy_u_mean
    copy_e_mean

GSDDMM functions
----------------

Generalized Sampled Dense-Dense Matrix Multiplication.
It computes edge features by add/sub/mul/div/dot features on source/destination
nodes or edges.

Like GSpMM, our implementation supports tensors on CPU/GPU in
PyTorch/MXNet/Tensorflow as input. All operators are equipped with autograd and
broadcasting.

The memory cost of GSDDMM is :math:`O(|E|D)` where :math:`|E|` refers to the number
of edges in the graph while :math:`D` refers to the feature size.

Note that we support ``dot`` operator, which semantically is the same as reduce
the last dimension by sum to the result of ``mul`` operator. However, the ``dot``
is more memory efficient because it *fuses* ``mul`` and sum reduction, which is
critical in the cases while the feature size on last dimension is non-trivial
(e.g. multi-head attention in Transformer-like models).

The following is an example showing how GSDDMM works:

   >>> import dgl
   >>> import torch as th
   >>> import dgl.ops as F
   >>> g = dgl.graph(([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]))  # 3 nodes, 6 edges
   >>> x = th.ones(3, 2, requires_grad=True)
   >>> x
   tensor([[1., 1.],
           [1., 1.],
           [1., 1.]], requires_grad=True)
   >>> y = th.arange(1, 7).float().view(3, 2).requires_grad_()
   >>> y
   tensor([[1., 2.],
           [3., 4.],
           [5., 6.]], requires_grad=True)
   >>> e = th.ones(6, 1, 2, requires_grad=True) * 2
   tensor([[[2., 2.]],
           [[2., 2.]],
           [[2., 2.]],
           [[2., 2.]],
           [[2., 2.]],
           [[2., 2.]]], grad_fn=<MulBackward0>)
   >>> out1 = F.u_div_v(g, x, y)
   tensor([[1.0000, 0.5000],
           [0.3333, 0.2500],
           [0.2000, 0.1667],
           [0.3333, 0.2500],
           [0.2000, 0.1667],
           [0.2000, 0.1667]], grad_fn=<GSDDMMBackward>)
   >>> out1.sum().backward()
   >>> x.grad
   tensor([[1.5333, 0.9167],
           [0.5333, 0.4167],
           [0.2000, 0.1667]])
   >>> y.grad
   tensor([[-1.0000, -0.2500],
           [-0.2222, -0.1250],
           [-0.1200, -0.0833]])
   >>> out2 = F.e_sub_v(g, e, y)
   >>> out2
   tensor([[[ 1.,  0.]],
           [[-1., -2.]],
           [[-3., -4.]],
           [[-1., -2.]],
           [[-3., -4.]],
           [[-3., -4.]]], grad_fn=<GSDDMMBackward>)
   >>> out3 = F.copy_v(g, y)
   >>> out3
   tensor([[1., 2.],
           [3., 4.],
           [5., 6.],
           [3., 4.],
           [5., 6.],
           [5., 6.]], grad_fn=<GSDDMMBackward>)
   >>> out4 = F.u_dot_v(g, x, y)
   >>> out4  # the last dimension was reduced to size 1.
   tensor([[ 3.],
           [ 7.],
           [11.],
           [ 7.],
           [11.],
           [11.]], grad_fn=<GSDDMMBackward>)

.. autosummary::
    :toctree: ../../generated/

    gsddmm
    u_add_v
    u_sub_v
    u_mul_v
    u_dot_v
    u_div_v
    u_add_e
    u_sub_e
    u_mul_e
    u_dot_e
    u_div_e
    e_add_v
    e_sub_v
    e_mul_v
    e_dot_v
    e_div_v
    v_add_u
    v_sub_u
    v_mul_u
    v_dot_u
    v_div_u
    e_add_u
    e_sub_u
    e_mul_u
    e_dot_u
    e_div_u
    v_add_e
    v_sub_e
    v_mul_e
    v_dot_e
    v_div_e
    copy_u
    copy_v

Like GSpMM, GSDDMM operators support both homogeneous and bipartite graph.

Segment Reduce Module
---------------------

DGL provide operators to reduce value tensor along the first dimension by segments.

.. autosummary::
   :toctree: ../../generated/

   segment_reduce

GatherMM and SegmentMM Module
-----------------------------

SegmentMM: DGL provide operators to perform matrix multiplication according to segments.

GatherMM: DGL provide operators to gather data according to the given indices and perform matrix multiplication.

.. autosummary::
   :toctree: ../../generated/

   gather_mm
   segment_mm

Supported Data types
--------------------
Operators defined in ``dgl.ops`` support floating point data types, i.e. the operands
must be ``half`` (``float16``) /``float``/``double`` tensors.
The input tensors must have the same data type (if one input tensor has type float16
and the other input tensor has data type float32, user must convert one of them to
align with the other one).

``float16`` data type support is disabled by default as it has a minimum GPU
compute capacity requirement of ``sm_53`` (Pascal, Volta, Turing and Ampere
architectures).

User can enable float16 for mixed precision training by compiling DGL from source
(see :doc:`Mixed Precision Training </guide/mixed_precision>` tutorial for details).

Relation with Message Passing APIs
----------------------------------

``dgl.update_all`` and ``dgl.apply_edges`` calls with built-in message/reduce functions
would be dispatched into function calls of operators defined in ``dgl.ops``:

    >>> import dgl
    >>> import torch as th
    >>> import dgl.ops as F
    >>> import dgl.function as fn
    >>> g = dgl.rand_graph(100, 1000)   # create a DGLGraph with 100 nodes and 1000 edges.
    >>> x = th.rand(100, 20)            # node features.
    >>> e = th.rand(1000, 20)
    >>>
    >>> # dgl.update_all + builtin functions
    >>> g.srcdata['x'] = x              # srcdata is the same as ndata for graphs with one node type.
    >>> g.edata['e'] = e
    >>> g.update_all(fn.u_mul_e('x', 'e', 'm'), fn.sum('m', 'y'))
    >>> y = g.dstdata['y']              # dstdata is the same as ndata for graphs with one node type.
    >>>
    >>> # use GSpMM operators defined in dgl.ops directly
    >>> y = F.u_mul_e_sum(g, x, e)
    >>>
    >>> # dgl.apply_edges + builtin functions
    >>> g.srcdata['x'] = x
    >>> g.dstdata['y'] = y
    >>> g.apply_edges(fn.u_dot_v('x', 'y', 'z'))
    >>> z = g.edata['z']
    >>>
    >>> # use GSDDMM operators defined in dgl.ops directly
    >>> z = F.u_dot_v(g, x, y)

It up to user to decide whether to use message-passing APIs or GSpMM/GSDDMM operators, and both
of them have the same efficiency. Programs written in message-passing APIs look more like DGL-style
but in some cases calling GSpMM/GSDDMM operators is more concise.

Note that on PyTorch all operators defined in ``dgl.ops`` support higher-order gradients, so as
message passing APIs because they entirely depend on these operators.


