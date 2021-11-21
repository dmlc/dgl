"""Torch modules for link prediction/knowledge graph completion."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch
import torch.nn as nn

__all__ = ['TransE', 'TransR']

class TransE(nn.Module):
    def __init__(self, num_rels, feats, p=1):
        super(TransE, self).__init__()

        self.rel_emb = nn.Embedding(num_rels, feats)
        self.p = p

    def reset_parameters(self):
        self.rel_emb.reset_parameters()

    def forward(self, h_head, h_tail, rels):
        h_rel = self.rel_emb(rels)

        return - torch.norm(h_head + h_rel - h_tail, p=self.p, dim=-1)

class TransR(nn.Module):
    def __init__(self, num_rels, rfeats, nfeats, p=1):
        super(TransR, self).__init__()

        self.rel_emb = nn.Embedding(num_rels, rfeats)
        self.rel_project = nn.Embedding(num_rels, nfeats * rfeats)
        self.rfeats = rfeats
        self.nfeats = nfeats
        self.p = p

    def reset_parameters(self):
        self.rel_emb.reset_parameters()
        self.rel_project.reset_parameters()

    def forward(self, h_head, h_tail, rels):
        h_rel = self.rel_emb(rels)
        proj_rel = self.rel_project(rels).reshape(-1, self.nfeats, self.rfeats)
        h_head = torch.einsum('ab,abc->ac', h_head, proj_rel)
        h_tail = torch.einsum('ab,abc->ac', h_tail, proj_rel)

        return - torch.norm(h_head + h_rel - h_tail, p=self.p, dim=-1)
