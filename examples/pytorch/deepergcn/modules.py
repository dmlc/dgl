import torch
import torch.nn as nn
import torch.nn.functional as F


def act_layer(act_type, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act_type.lower()
    
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act)
    
    return layer


def norm_layer(norm_type, nc):
    norm = norm_type.lower()

    if norm == 'batch':
        layer = nn.BatchNorm1d(nc, affine=True)
    elif norm == 'layer':
        layer = nn.LayerNorm(nc, elementwise_affine=True)
    elif norm == 'instance':
        layer = nn.InstanceNorm1d(nc, affine=False)
    else:
        raise NotImplementedError(f'Normalization layer {norm} is not supported.')

    return layer


class MLP(nn.Sequential):
    r"""

    Description
    -----------
    From equation (5) in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_
    """
    def __init__(self,
                 channels,
                 act='relu',
                 norm=None,
                 dropout=0.,
                 bias=True):
        layers = []
        
        for i in range(1, len(channels)):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias))
            if i < len(channels) - 1:
                if norm is not None and norm.lower() != 'none':
                    layers.append(norm_layer(norm, channels[i]))
                if act is not None and act.lower() != 'none':
                    layers.append(act_layer(act))
                layers.append(nn.Dropout(dropout))
        
        super(MLP, self).__init__(*layers)


class MessageNorm(nn.Module):
    r"""
    
    Description
    -----------
    Message normalization was introduced in `DeeperGCN: All You Need to Train Deeper GCNs <https://arxiv.org/abs/2006.07739>`_

    Parameters
    ----------
    learn_scale: bool
        Whether s is a learnable scaling factor or not. Default is False.
    """
    def __init__(self, learn_scale=False):
        super(MessageNorm, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=learn_scale)

    def forward(self, feats, msg, p=2):
        msg = F.normalize(msg, p=2, dim=-1)
        feats_norm = feats.norm(p=p, dim=-1, keepdim=True)
        return msg * feats_norm * self.scale
