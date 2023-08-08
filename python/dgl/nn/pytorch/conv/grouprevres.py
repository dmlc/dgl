"""Torch module for grouped reversible residual connections for GNNs"""
# pylint: disable= no-member, arguments-differ, invalid-name, C0116, R1728
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn


class InvertibleCheckpoint(torch.autograd.Function):
    r"""Extension of torch.autograd"""

    @staticmethod
    def forward(ctx, fn, fn_inverse, num_inputs, *inputs_and_weights):
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = inputs_and_weights[num_inputs:]
        inputs = inputs_and_weights[:num_inputs]
        ctx.input_requires_grad = []

        with torch.no_grad():
            # Make a detached copy, which shares the storage
            x = []
            for element in inputs:
                if isinstance(element, torch.Tensor):
                    x.append(element.detach())
                    ctx.input_requires_grad.append(element.requires_grad)
                else:
                    x.append(element)
                    ctx.input_requires_grad.append(None)
            # Detach the output, which then allows discarding the intermediary results
            outputs = ctx.fn(*x).detach_()

        # clear memory of input node features
        inputs[1].untyped_storage().resize_(0)

        # store for backward pass
        ctx.inputs = [inputs]
        ctx.outputs = [outputs]

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "InvertibleCheckpoint is not compatible with .grad(), \
                               please use .backward() if possible"
            )
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError(
                "Trying to perform backward on the InvertibleCheckpoint \
                               for more than once."
            )
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # reconstruct input node features
        with torch.no_grad():
            # inputs[0] is DGLGraph and inputs[1] is input node features
            inputs_inverted = ctx.fn_inverse(
                *((inputs[0], outputs) + inputs[2:])
            )
            # clear memory of outputs
            outputs.untyped_storage().resize_(0)

            x = inputs[1]
            x.untyped_storage().resize_(int(np.prod(x.size())))
            x.set_(inputs_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            detached_inputs = []
            for i, element in enumerate(inputs):
                if isinstance(element, torch.Tensor):
                    element = element.detach()
                    element.requires_grad = ctx.input_requires_grad[i]
                detached_inputs.append(element)

            detached_inputs = tuple(detached_inputs)
            temp_output = ctx.fn(*detached_inputs)

        filtered_detached_inputs = tuple(
            filter(
                lambda x: getattr(x, "requires_grad", False), detached_inputs
            )
        )
        gradients = torch.autograd.grad(
            outputs=(temp_output,),
            inputs=filtered_detached_inputs + ctx.weights,
            grad_outputs=grad_outputs,
        )

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights) :]

        return (None, None, None) + gradients


class GroupRevRes(nn.Module):
    r"""Grouped reversible residual connections for GNNs, as introduced in
    `Training Graph Neural Networks with 1000 Layers <https://arxiv.org/abs/2106.07476>`__

    It uniformly partitions an input node feature :math:`X` into :math:`C` groups
    :math:`X_1, X_2, \cdots, X_C` across the channel dimension. Besides, it makes
    :math:`C` copies of the input GNN module :math:`f_{w1}, \cdots, f_{wC}`. In the
    forward pass, each GNN module only takes the corresponding group of node features.

    The output node representations :math:`X^{'}` are computed as follows.

    .. math::

        X_0^{'} = \sum_{i=2}^{C}X_i

        X_i^{'} = f_{wi}(X_{i-1}^{'}, g, U) + X_i, i\in\{1,\cdots,C\}

        X^{'} = X_1^{'} \, \Vert \, \ldots \, \Vert \, X_C^{'}

    where :math:`g` is the input graph, :math:`U` is arbitrary additional input arguments like
    edge features, and :math:`\, \Vert \,` is concatenation.

    Parameters
    ----------
    gnn_module : nn.Module
        GNN module for message passing. :attr:`GroupRevRes` will clone the module for
        :attr:`groups`-1 number of times, yielding :attr:`groups` copies in total.
        The input and output node representation size need to be the same. Its forward
        function needs to take a DGLGraph and the associated input node features in order,
        optionally followed by additional arguments like edge features.
    groups : int, optional
        The number of groups.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> import torch.nn as nn
    >>> from dgl.nn import GraphConv, GroupRevRes

    >>> class GNNLayer(nn.Module):
    ...     def __init__(self, feats, dropout=0.2):
    ...         super(GNNLayer, self).__init__()
    ...         # Use BatchNorm and dropout to prevent gradient vanishing
    ...         # In particular if you use a large number of GNN layers
    ...         self.norm = nn.BatchNorm1d(feats)
    ...         self.conv = GraphConv(feats, feats)
    ...         self.dropout = nn.Dropout(dropout)
    ...
    ...     def forward(self, g, x):
    ...         x = self.norm(x)
    ...         x = self.dropout(x)
    ...         return self.conv(g, x)

    >>> num_nodes = 5
    >>> num_edges = 20
    >>> feats = 32
    >>> groups = 2
    >>> g = dgl.rand_graph(num_nodes, num_edges)
    >>> x = torch.randn(num_nodes, feats)
    >>> conv = GNNLayer(feats // groups)
    >>> model = GroupRevRes(conv, groups)
    >>> out = model(g, x)
    """

    def __init__(self, gnn_module, groups=2):
        super(GroupRevRes, self).__init__()
        self.gnn_modules = nn.ModuleList()
        for i in range(groups):
            if i == 0:
                self.gnn_modules.append(gnn_module)
            else:
                self.gnn_modules.append(deepcopy(gnn_module))
        self.groups = groups

    def _forward(self, g, x, *args):
        xs = torch.chunk(x, self.groups, dim=-1)

        if len(args) == 0:
            args_chunks = [()] * self.groups
        else:
            chunked_args = list(
                map(lambda arg: torch.chunk(arg, self.groups, dim=-1), args)
            )
            args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.groups):
            y_in = xs[i] + self.gnn_modules[i](g, y_in, *args_chunks[i])
            ys.append(y_in)

        out = torch.cat(ys, dim=-1)

        return out

    def _inverse(self, g, y, *args):
        ys = torch.chunk(y, self.groups, dim=-1)

        if len(args) == 0:
            args_chunks = [()] * self.groups
        else:
            chunked_args = list(
                map(lambda arg: torch.chunk(arg, self.groups, dim=-1), args)
            )
            args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.groups - 1, -1, -1):
            if i != 0:
                y_in = ys[i - 1]
            else:
                y_in = sum(xs)

            x = ys[i] - self.gnn_modules[i](g, y_in, *args_chunks[i])
            xs.append(x)

        x = torch.cat(xs[::-1], dim=-1)

        return x

    def forward(self, g, x, *args):
        r"""Apply the GNN module with grouped reversible residual connection.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : torch.Tensor
            The input feature of shape :math:`(N, D_{in})`, where :math:`D_{in}` is size
            of input feature, :math:`N` is the number of nodes.
        args
            Additional arguments to pass to :attr:`gnn_module`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{in})`.
        """
        args = (g, x) + args
        y = InvertibleCheckpoint.apply(
            self._forward,
            self._inverse,
            len(args),
            *(args + tuple([p for p in self.parameters() if p.requires_grad]))
        )

        return y
