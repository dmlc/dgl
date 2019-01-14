import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

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
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.size = size
        self.padding_idx = padding_idx
        self.smoothing = smoothing

    def forward(self, x, target):
        # x: (*, n_classes)
        # target: (*)
        assert x.size(1) == self.size
        with T.no_grad():
            tgt_dist = T.zeros_like(x, dtype=T.float)
            tgt_dist.fill_(self.smoothing / (self.size - 2)) # one for padding, another for label
            tgt_dist[:, self.padding_idx] = 0
            tgt_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

            mask = T.nonzero(target == self.padding_idx)
            if mask.shape[0] > 0:
                tgt_dist.index_fill_(0, mask.squeeze(), 0)

        return self.criterion(x, tgt_dist)

class SimpleLossCompute(nn.Module):
    eps=1e-8
    def __init__(self, criterion, opt=None):
        """
        opt is required during training
        """
        super(SimpleLossCompute, self).__init__()
        self.criterion = criterion
        self.opt = opt
        self.acc_loss = 0
        self.n_correct = 0
        self.norm_term = 0
        self.loss = 0

    @property
    def avg_loss(self):
        return (self.acc_loss + self.eps) / (self.norm_term + self.eps)

    @property
    def accuracy(self):
        return (self.n_correct + self.eps) / (self.norm_term + self.eps)

    def backward(self):
        self.loss.backward()

    def __call__(self, y_pred, y, norm, is_train=True):
        y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y.contiguous().view(-1)
        self.loss = self.criterion(
            y_pred, y
        ) / norm
        if is_train:
            self.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        self.n_correct += ((y_pred.max(dim=-1)[1] == y) & (y != self.criterion.padding_idx)).sum().item()
        self.acc_loss += self.loss.item() * norm
        self.norm_term += norm
        return self.loss.item() * norm

class MultiGPULossCompute(SimpleLossCompute):
    def __init__(self, criterion, dev_id, ndev, model, opt=None):
        super(MultiGPULossCompute, self).__init__(criterion, opt)
        self.dev_id = dev_id
        self.ndev = ndev
        self.model = model

    def backward(self):
        # multi-gpu synchronous backward
        self.loss.backward()
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.ndev
