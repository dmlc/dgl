"""Torch module for grouped reversible residual connections for GNNs"""
# pylint: disable= no-member, arguments-differ, invalid-name
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

class InvertibleCheckpoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, fn_inverse, inputs, weights):
        # store in context
        ctx.fn = fn
        ctx.fn_inverse = fn_inverse
        ctx.weights = weights
        g, x = inputs
        ctx.input_requires_grad = x.requires_grad

        with torch.no_grad():
            output = ctx.fn(g, x.detach())
        # Detach y in-place (computations in between the input and output can now be discarded)
        output = output.detach_()

        # clear memory of node features
        x.storage().resize_(0)

        # store these tensor nodes for backward pass
        ctx.inputs = [inputs]
        ctx.outputs = [output]

        return output

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpoint is not compatible with .grad(), \
                               please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward for more than once.")
        inputs = ctx.inputs.pop()
        output = ctx.outputs.pop()
        g, x = inputs

        # recompute input
        with torch.no_grad():
            x_inverted = ctx.fn_inverse(g, output)
            output.storage().resize_(0)

            x.storage().resize_(int(np.prod(x.size())))
            x.set_(x_inverted)

        # compute gradients
        with torch.set_grad_enabled(True):
            x = x.detach()
            x.requires_grad = ctx.input_requires_grad
            tmp_output = ctx.fn(g, x)

        filtered_inputs = (x,) if x.requires_grad else ()
        gradients = torch.autograd.grad(outputs=(tmp_output,),
                                        inputs=filtered_inputs + ctx.weights,
                                        grad_outputs=grad_outputs)

        # None for g
        input_gradients = [None]
        if x.requires_grad:
            input_gradients.append(gradients[0])
        else:
            input_gradients.append(None)
        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        # None for fn, fn_inverse
        return (None, None) + gradients

class GroupRevRes(nn.Module):
    r"""Grouped reversible residual connections for GNNs, as introduced in
    `Training Graph Neural Networks with 1000 Layers <https://arxiv.org/abs/2106.07476>`__

    Parameters
    ----------
    gnn_module : nn.Module
        GNN module for message passing.
    group : int, optional
        The input node features will be partitioned into the specified number of groups.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> from dgl.nn import GraphConv, GroupRevRes

    >>> num_nodes = 5
    >>> num_edges = 20
    >>> in_feats = 32
    >>> out_feats = 64
    >>> g = dgl.rand_graph(num_nodes, num_edges)
    >>> h = torch.randn(num_nodes, in_feats)
    >>> conv = GraphConv(in_feats, out_feats)
    >>> model = GroupRevRes(conv)
    >>> out = model(g, h)
    """
    def __init__(self, gnn_module, group=2):
        self.gnn_modules = nn.ModuleList()
        self.group = group
        for i in range(group):
            if i == 0:
                self.gnn_modules.append(gnn_module)
            else:
                self.gnn_modules.append(deepcopy(gnn_module))

    def _forward(self, g, x):
        x_chunks = torch.chunk(x, self.group, dim=-1)
        y_in = sum(x_chunks[1:])

        y_chunks = []
        for i in range(self.group):
            tmp_out = self.gnn_modules[i](g, y_in)
            y = x_chunks[i] + tmp_out
            y_in = y
            y_chunks.append(y)

        out = torch.cat(y_chunks, dim=-1)

        return out

    def _inverse(self, g, y):
        y_chunks = torch.chunk(y, self.group, dim=-1)

        x_chunks = []
        # self.group-1, ..., 0
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = y_chunks[i-1]
            else:
                y_in = sum(x_chunks)

            tmp_out = self.gnn_modules[i](g, y_in)
            x = y_chunks[i] - tmp_out
            x_chunks.append(x)

        x = torch.cat(x_chunks[::-1], dim=-1)

        return x

    def forward(self, g, x):
        y = InvertibleCheckpoint.apply(
            self._forward,
            self._inverse,
            (g, x),
            tuple([p for p in self.parameters() if p.requires_grad])
        )
        if isinstance(y, tuple) and len(y) == 1:
            return y[0]
        return y
