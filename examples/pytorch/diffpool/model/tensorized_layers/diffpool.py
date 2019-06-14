import torch
from torch import nn as nn

from model.layers.assignment import DiffPoolAssignment, RoutingAssignment, MultihopDiffPoolAssignment, GlobalInfoDiffPoolAssignment
from model.layers.graphsage import BatchedGraphSAGE


# from model.layers.loss.link_pred import LinkPredLoss


class EntropyLoss(nn.Module):
    # Return Scalar
    def forward(self, adj, anext, s_l, mask):
        if mask is not None:
            s_l = s_l + 1 * ((1 - mask).unsqueeze(-1))
            entropy = (torch.distributions.Categorical(probs=s_l).entropy() * mask).sum(-1).mean(-1)
            assert not torch.isnan(entropy)
        else:
            entropy = (torch.distributions.Categorical(probs=s_l).entropy()).sum(-1).mean(-1)
            assert not torch.isnan(entropy)
        return entropy


class MinCutLoss(nn.Module):
    # Return Scalar
    # anext with shape [batch, batched_node_num, batched_node_num]
    def forward(self, adj, anext, s_l, mask):
        mask = torch.ones_like(anext) - torch.eye(anext.size(1)).to(anext.device)
        min_cut_loss = (mask * anext).mean(0).sum()
        return min_cut_loss


class NCutLoss(nn.Module):
    # Return Scalar
    # anext with shape [batch, batched_node_num, batched_node_num]
    def forward(self, adj, anext, s_l, mask):
        degree = anext.sum(-1, keepdim=True)
        diag_mask = torch.ones_like(anext) - torch.eye(anext.size(1)).to(anext.device)
        ncut = ((diag_mask * anext) / (degree + 1e-10)).mean(0).sum()
        return ncut

class LinkPredLoss(nn.Module):

    def forward(self, adj, anext, s_l, mask):
        if mask is not None:
            s_l = s_l * mask.unsqueeze(-1)
        link_pred_loss = (adj - s_l.matmul(s_l.transpose(-1, -2))).norm(dim=(1, 2))
        link_pred_loss = link_pred_loss / (adj.size(1) * adj.size(2))
        return link_pred_loss.mean()

class BatchedDiffPool(nn.Module):
    def __init__(self, nfeat, nnext, nhid, link_pred=False, gumbel=False,
                 min_cut=False, ncut=False, entropy=False, rtn=-1):
        super(BatchedDiffPool, self).__init__()
        self.link_pred = link_pred
        self.log = {}
        self.min_cut = True
        self.link_pred_layer = LinkPredLoss()
        self.embed = BatchedGraphSAGE(nfeat, nhid, use_bn=True)
        if rtn > 0:
            self.assign = RoutingAssignment(nfeat, nnext, gumbel)
        else:
            #self.assign = MultihopDiffPoolAssignment(nfeat, nnext, gumbel)
            #self.assign = DiffPoolAssignment(nfeat, nnext, gumbel)
            self.assign = GlobalInfoDiffPoolAssignment(nfeat, nnext, gumbel)
        self.reg_loss = nn.ModuleList([])
        self.loss_log = {}
        if link_pred:
            self.reg_loss.append(LinkPredLoss())
        if entropy:
            self.reg_loss.append(EntropyLoss())
        if ncut:
            self.reg_loss.append(NCutLoss())


    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = self.assign(x, adj, mask)
        if mask is not None:
            s_l = s_l * mask.unsqueeze(-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        anext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)

        for loss_layer in self.reg_loss:
            loss_name = str(type(loss_layer).__name__)
            self.loss_log[loss_name] = loss_layer(adj, anext, s_l, mask)
        if log:
            self.log['a'] = anext.cpu().numpy()
        return xnext, anext


