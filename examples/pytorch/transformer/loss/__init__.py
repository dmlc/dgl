import torch as T
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    Computer loss at one time step.
    """

    def __init__(self, size, padding_idx, smoothing=0.0):
        """Label Smoothing module
        args:
            size: vocab_size
            padding_idx: index for symbol `padding`
            smoothing: smoothing ratio
        """
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (*, n_classes)
        # target: (*)
        assert x.size(1) == self.size
        with T.no_grad():
            tgt_dist = T.zeros_like(x, dtype=T.float)
            tgt_dist.fill_(
                self.smoothing / (self.size - 2)
            )  # one for padding, another for label
            tgt_dist[:, self.padding_idx] = 0
            tgt_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

            mask = T.nonzero(target == self.padding_idx)
            if mask.shape[0] > 0:
                tgt_dist.index_fill_(0, mask.squeeze(), 0)

        return self.criterion(x, tgt_dist)


class SimpleLossCompute(nn.Module):
    eps = 1e-8

    def __init__(self, criterion, grad_accum, opt=None):
        """Loss function and optimizer for single device

        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        grad_accum: int
            number of batches to accumulate gradients
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(SimpleLossCompute, self).__init__()
        self.criterion = criterion
        self.opt = opt
        self.acc_loss = 0
        self.n_correct = 0
        self.norm_term = 0
        self.loss = 0
        self.batch_count = 0
        self.grad_accum = grad_accum

    def __enter__(self):
        self.batch_count = 0

    def __exit__(self, type, value, traceback):
        # if not enough batches accumulated and there are gradients not applied,
        # do one more step
        if self.batch_count > 0:
            self.step()

    @property
    def avg_loss(self):
        return (self.acc_loss + self.eps) / (self.norm_term + self.eps)

    @property
    def accuracy(self):
        return (self.n_correct + self.eps) / (self.norm_term + self.eps)

    def step(self):
        self.opt.step()
        self.opt.optimizer.zero_grad()

    def backward_and_step(self):
        self.loss.backward()
        self.batch_count += 1
        # accumulate self.grad_accum times then synchronize and update
        if self.batch_count == self.grad_accum:
            self.step()
            self.batch_count = 0

    def __call__(self, y_pred, y, norm):
        y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y.contiguous().view(-1)
        self.loss = self.criterion(y_pred, y) / norm
        if self.opt is not None:
            self.backward_and_step()
        self.n_correct += (
            ((y_pred.max(dim=-1)[1] == y) & (y != self.criterion.padding_idx))
            .sum()
            .item()
        )
        self.acc_loss += self.loss.item() * norm
        self.norm_term += norm
        return self.loss.item() * norm


class MultiGPULossCompute(SimpleLossCompute):
    def __init__(self, criterion, ndev, grad_accum, model, opt=None):
        """Loss function and optimizer for multiple devices

        Parameters
        ----------
        criterion: torch.nn.Module
            criterion to compute loss
        ndev: int
            number of devices used
        grad_accum: int
            number of batches to accumulate gradients
        model: torch.nn.Module
            model to optimizer (needed to iterate and synchronize all parameters)
        opt: Optimizer
            Model optimizer to use. If None, then no backward and update will be
            performed
        """
        super(MultiGPULossCompute, self).__init__(
            criterion, grad_accum, opt=opt
        )
        self.ndev = ndev
        self.model = model

    def step(self):
        # multi-gpu synchronize gradients
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.ndev
        self.opt.step()
        self.opt.optimizer.zero_grad()
