import torch as T
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
        self.reset()

    @property
    def avg_loss(self):
        return (self.acc_loss + self.eps) / (self.norm_term + self.eps)

    @property
    def accuracy(self):
        return (self.n_correct + self.eps) / (self.norm_term + self.eps)

    def reset(self):
        self.acc_loss = 0
        self.n_correct = 0
        self.norm_term = 0

    def __call__(self, y_pred, y, norm):
        y_pred = y_pred.contiguous().view(-1, y_pred.shape[-1])
        y = y.contiguous().view(-1)
        loss = self.criterion(
            y_pred, y
        ) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()
        self.n_correct += ((y_pred.max(dim=-1)[1] == y) & (y != self.criterion.padding_idx)).sum().item()
        self.acc_loss += loss.item() * norm
        self.norm_term += norm
        return loss.item() * norm

class MultiGPULossCompute(SimpleLossCompute):
    def __init__(self, criterion, devices, opt=None, chunk_size=5):
        self.criterion = criterion
        self.opt = opt
        self.devices = devices
        self.chunk_size = chunk_size

    def __call__(self, y_preds, ys, norms):
        pass
