.. _guide-mixed_precision:

Chapter 8: Mixed Precision Training
===================================
DGL is compatible with `PyTorch's automatic mixed precision package
<https://pytorch.org/docs/stable/amp.html>`_
for mixed precision training, thus saving both training time and GPU memory
consumption. To enable this feature, users need to install PyTorch 1.6+ with python 3.7+ and
build DGL from source file to support ``float16`` data type (this feature is
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
DGL with fp16 support allows message-passing on ``float16`` features for both
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
DGL relies on PyTorch's AMP package for mixed precision training,
and the user experience is exactly
the same as `PyTorch's <https://pytorch.org/docs/stable/notes/amp_examples.html>`_.

By wrapping the forward pass (including loss computation) of your GNN model with
``torch.cuda.amp.autocast()``, PyTorch automatically selects the appropriate datatype
for each op and tensor. Half precision tensors are memory efficient, most operators
on half precision tensors are faster as they leverage GPU's tensorcores.

Small Gradients in ``float16`` format have underflow problems (flush to zero), and
PyTorch provides a ``GradScaler`` module to address this issue. ``GradScaler`` multiplies
loss by a factor and invokes backward pass on scaled loss, and unscales graidents before
optimizers update the parameters, thus preventing the underflow problem.
The scale factor is determined automatically.

Following is the training script of 3-layer GAT on Reddit dataset (w/ 114 million edges),
note the difference in codes when ``use_fp16`` is activated/not activated:

.. code::

    import torch 
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.cuda.amp import autocast, GradScaler
    import dgl
    from dgl.data import RedditDataset
    from dgl.nn import GATConv

    use_fp16 = True


    class GAT(nn.Module):
        def __init__(self,
                     in_feats,
                     n_hidden,
                     n_classes,
                     heads):
            super().__init__()
            self.layers = nn.ModuleList()
            self.layers.append(GATConv(in_feats, n_hidden, heads[0], activation=F.elu))
            self.layers.append(GATConv(n_hidden * heads[0], n_hidden, heads[1], activation=F.elu))
            self.layers.append(GATConv(n_hidden * heads[1], n_classes, heads[2], activation=F.elu))

        def forward(self, g, h):
            for l, layer in enumerate(self.layers):
                h = layer(g, h)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                else:
                    h = h.mean(1)
            return h

    # Data loading
    data = RedditDataset()
    device = torch.device(0)
    g = data[0]
    g = dgl.add_self_loop(g)
    g = g.int().to(device)
    train_mask = g.ndata['train_mask']
    features = g.ndata['feat']
    labels = g.ndata['label']
    in_feats = features.shape[1]
    n_hidden = 256
    n_classes = data.num_classes
    n_edges = g.number_of_edges()
    heads = [1, 1, 1]
    model = GAT(in_feats, n_hidden, n_classes, heads)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    # Create gradient scaler
    scaler = GradScaler()

    for epoch in range(100):
        model.train()
        optimizer.zero_grad()

        # Wrap forward pass with autocast
        with autocast(enabled=use_fp16):
            logits = model(g, features)
            loss = F.cross_entropy(logits[train_mask], labels[train_mask])
        
        if use_fp16:
            # Backprop w/ gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        print('Epoch {} | Loss {}'.format(epoch, loss.item()))


On a NVIDIA V100 (16GB) machine, training this model without fp16 consumes
15.2GB GPU memory; with fp16 turned on, the training consumes 12.8G
GPU memory, the loss converges to similar values in both settings.
If we change the number of heads to ``[2, 2, 2]``, training without fp16
triggers GPU OOM(out-of-memory) issue while training with fp16 consumes
15.7G GPU memory.

DGL is still improving its half-precision support and the compute kernel's
performance is far from optimal, please stay tuned to our future updates.
