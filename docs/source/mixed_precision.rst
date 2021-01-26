Mixed Precision Training
========================
DGL is compatible with `PyTorch's automatic mixed precision package
<https://pytorch.org/docs/stable/amp.html>`_
for mixed precision training, thus saving both training time and GPU memory
consumption. To enable this feature, user need to install PyTorch 1.6+ and
build DGL from source file to support float16 data type (this feature is
still in its beta stage and we do not provide official pre-built pip wheels).

Installation
------------
First download DGL's source code from GitHub and build the shared library
with flag ``USE_FP16=ON``.

.. code:: bash

   git clone --recurse-submodules https://github.com/dmlc/dgl.git
   cd dgl
   mkdir build
   cd build
   cmake -DUSE_CUDA=ON -DUSE_FP16=ON ..
   make -j

Then install the Python binding.

.. code:: bash

   cd ../python
   python setup.py install

Message-Passing with Half Precision
-----------------------------------
DGL with fp16 support allows message-passing on float16 features for both
UDF(User Defined Function)s and built-in functions (e.g. ``dgl.function.sum``,
``dgl.function.copy_u``).

The following examples shows how to use DGL's message-passing API on half-precision
features:

    >>> import torch
    >>> import dgl
    >>> import dgl.function as fn
    >>> g = dgl.rand_graph(30, 100).to(0)  # Create a graph on GPU w/ 30 nodes and 100 edges.
    >>> g.ndata['h'] = torch.rand(30, 16).to(0).half()  # Create fp16 node features.
    >>> g.edata['w'] = torch.rand(100, 1).to(0).half()  # Create fp16 edge features.
    >>> # Use DGL's built-in functions for message passing on fp16 features.
    >>> g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum('m', 'x'))
    >>> g.ndata['x'][0]
    tensor([0.3391, 0.2208, 0.7163, 0.6655, 0.7031, 0.5854, 0.9404, 0.7720, 0.6562,
            0.4028, 0.6943, 0.5908, 0.9307, 0.5962, 0.7827, 0.5034],
           device='cuda:0', dtype=torch.float16)
    >>> g.apply_edges(fn.u_dot_v('h', 'x', 'hx'))
    >>> g.edata['hx'][0]
    tensor([5.4570], device='cuda:0', dtype=torch.float16)
    >>> # Use UDF(User Defined Functions) for message passing on fp16 features.
    >>> def message(edges):
    ...     return {'m': edges.src['h'] * edges.data['w']}
    ...
    >>> def reduce(nodes):
    ...     return {'y': torch.sum(nodes.mailbox['m'], 1)}
    ...
    >>> def dot(edges):
    ...     return {'hy': (edges.src['h'] * edges.dst['y']).sum(-1, keepdims=True)}
    ...
    >>> g.update_all(message, reduce)
    >>> g.ndata['y'][0]
    tensor([0.3394, 0.2209, 0.7168, 0.6655, 0.7026, 0.5854, 0.9404, 0.7720, 0.6562,
            0.4028, 0.6943, 0.5908, 0.9307, 0.5967, 0.7827, 0.5039],
           device='cuda:0', dtype=torch.float16)
    >>> g.apply_edges(dot)
    >>> g.edata['hy'][0]
    tensor([5.4609], device='cuda:0', dtype=torch.float16)


End-to-End Mixed Precision Training
-----------------------------------
The user experience on end-to-end mixed precision training of DGL is exactly the same
as `PyTorch's <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.
