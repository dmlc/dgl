import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Sequential):
    r"""

    Description
    -----------
    From equation (5) in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"
    """

    def __init__(self, channels, act="relu", dropout=0.0, bias=True):
        layers = []

        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                layers.append(nn.BatchNorm1d(channels[i], affine=True))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*layers)


class MessageNorm(nn.Module):
    r"""

    Description
    -----------
    Message normalization was introduced in "DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>"

    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """

    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=learn_scale
        )

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale
