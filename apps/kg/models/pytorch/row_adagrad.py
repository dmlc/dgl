#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

from torch.optim import Optimizer


class RowAdagrad(Optimizer):
    """Implements a row-wise variant of the Adagrad algorithm.
    Assumes that all the model parameters are 2-dimensional tensors
    containing embedding weights.

    Code mostly copy-pasted from torch/optim/Adagrad
    """

    def __init__(self, params, lr=1e-2, lr_decay=0, weight_decay=0):
        defaults = dict(lr=lr, lr_decay=lr_decay, weight_decay=weight_decay)
        super().__init__(params, defaults)

        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                assert p.data.ndimension() == 2, (
                    "RowAdagrad only works on 2D parameter tensors")
                state['sum'] = p.data.new().resize_(p.data.size(0)).zero_()

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if grad.is_sparse:
                        raise RuntimeError("weight_decay option is not "
                                           "compatible with sparse gradients ")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1 + (state['step'] - 1) * group['lr_decay'])

                if grad.is_sparse:
                    if grad._indices().numel() == 0:
                        continue
                    # the update is non-linear so indices must be unique
                    grad = grad.coalesce()
                    grad_indices = grad._indices()[0]
                    grad_values = grad._values()

                    state['sum'].index_add_(0, grad_indices, (grad_values * grad_values).mean(1))
                    std = state['sum'][grad_indices]  # _sparse_mask
                    std_values = std.sqrt_().add_(1e-10).unsqueeze(1)
                    # print('std_values')
                    # print(std_values)
                    p.data.index_add_(0, grad_indices, -clr * grad_values / std_values)
                else:
                    state['sum'] += (grad * grad).mean(1)
                    std = state['sum'].sqrt().add_(1e-10)
                    p.data.addcdiv_(-clr, grad, std.unsqueeze(1))

        return loss
