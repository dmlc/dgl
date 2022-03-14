"""Torch module for grouped reversible residual connections for GNNs"""
# pylint: disable= no-member, arguments-differ, invalid-name
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
        inputs[0].storage().resize_(0)

        # store for backward pass
        ctx.inputs = [inputs]
        ctx.outputs = [outputs]

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError("InvertibleCheckpoint is not compatible with .grad(), \
                               please use .backward() if possible")
        # retrieve input and output tensor nodes
        if len(ctx.outputs) == 0:
            raise RuntimeError("Trying to perform backward on the InvertibleCheckpoint \
                               for more than once.")
        inputs = ctx.inputs.pop()
        outputs = ctx.outputs.pop()

        # reconstruct input node features
        with torch.no_grad():
            # inputs[0] used to have the input node features
            inputs_inverted = ctx.fn_inverse(*((outputs,)+inputs[1:]))
            # clear memory of outputs
            outputs.storage().resize_(0)

            x = inputs[0]
            x.storage().resize_(int(np.prod(x.size())))
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

        filtered_detached_inputs = tuple(filter(lambda x: x.requires_grad, detached_inputs))
        gradients = torch.autograd.grad(outputs=(temp_output,),
                                        inputs=filtered_detached_inputs + ctx.weights,
                                        grad_outputs=grad_outputs)

        input_gradients = []
        i = 0
        for rg in ctx.input_requires_grad:
            if rg:
                input_gradients.append(gradients[i])
                i += 1
            else:
                input_gradients.append(None)

        gradients = tuple(input_gradients) + gradients[-len(ctx.weights):]

        return (None, None, None) + gradients


class GroupRevRes(nn.Module):
    r"""Grouped reversible residual connections for GNNs, as introduced in
    `Training Graph Neural Networks with 1000 Layers <https://arxiv.org/abs/2106.07476>`__

    Parameters
    ----------
    gnn_module : nn.Module
        GNN module for message passing. The input and output node representation size need
        to be the same.
    group : int, optional
        The input node features will be partitioned into the specified number of groups.

    Examples
    --------

    >>> import dgl
    >>> import torch
    >>> from dgl.nn import GraphConv, GroupRevRes

    >>> num_nodes = 5
    >>> num_edges = 20
    >>> feats = 32
    >>> group = 2
    >>> g = dgl.rand_graph(num_nodes, num_edges)
    >>> h = torch.randn(num_nodes, feats)
    >>> conv = GraphConv(feats // group, feats // group)
    >>> model = GroupRevRes(conv, group)
    >>> out = model(g, h)
    """
    def __init__(self, gnn_module, group=2):
        super(GroupRevRes, self).__init__()
        self.gnn_modules = nn.ModuleList()
        for i in range(group):
            if i == 0:
                self.gnn_modules.append(gnn_module)
            else:
                self.gnn_modules.append(deepcopy(gnn_module))
        self.group = group

    def _forward(self, x, g, *args):
        xs = torch.chunk(x, self.group, dim=-1)

        if len(args) == 0:
            args_chunks = [()] * self.group
        else:
            chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
            args_chunks = list(zip(*chunked_args))
        y_in = sum(xs[1:])

        ys = []
        for i in range(self.group):
            tmp_out = self.gnn_modules[i](g, y_in, *args_chunks[i])
            y = xs[i] + tmp_out
            y_in = y
            ys.append(y)

        out = torch.cat(ys, dim=-1)

        return out

    def _inverse(self, y, g, *args):
        ys = torch.chunk(y, self.group, dim=-1)

        if len(args) == 0:
            args_chunks = [()] * self.group
        else:
            chunked_args = list(map(lambda arg: torch.chunk(arg, self.group, dim=-1), args))
            args_chunks = list(zip(*chunked_args))

        xs = []
        for i in range(self.group-1, -1, -1):
            if i != 0:
                y_in = ys[i-1]
            else:
                y_in = sum(xs)

            tmp_out = self.gnn_modules[i](g, y_in, *args_chunks[i])
            x = ys[i] - tmp_out
            xs.append(x)

        x = torch.cat(xs[::-1], dim=-1)

        return x

    def forward(self, g, x, *args):
        args = (x, g) + args
        y = InvertibleCheckpoint.apply(
            self._forward,
            self._inverse,
            len(args),
            *(args + tuple([p for p in self.parameters() if p.requires_grad])))

        return y
