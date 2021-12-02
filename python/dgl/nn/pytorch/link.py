"""Torch modules for link prediction/knowledge graph completion."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import torch
import torch.nn as nn

__all__ = ['TransE', 'TransR']

class TransE(nn.Module):
    r"""

    Description
    -----------
    Similarity measure introduced in `Translating Embeddings for Modeling Multi-relational Data
    <https://papers.nips.cc/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html>`__.
    Mathematically, it is defined as follows:

    .. math::

        - {\| h + r - t \|}_p

    where :math:`h` is the head embedding, :math:`r` is the relation embedding, and
    :math:`t` is the tail embedding.

    Parameters
    ----------
    num_rels : int
        Number of relation types.
    feats : int
        Embedding size.
    p : int, optional
        The p to use for Lp norm, which can be 1 or 2.

    Examples
    --------
    >>> import torch as th
    >>> from dgl.nn import TransE
    >>> scorer = TransE(num_rels=3, feats=4)
    """
    def __init__(self, num_rels, feats, p=1):
        super(TransE, self).__init__()

        self.rel_emb = nn.Embedding(num_rels, feats)
        self.p = p

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        self.rel_emb.reset_parameters()

    def forward(self, h_head, h_tail, rels):
        r"""

        Description
        -----------
        Score triples.

        Parameters
        ----------
        h_head : torch.Tensor
            Head entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        h_tail : torch.Tensor
            Tail entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        rels : torch.Tensor
            Relation types. It is a LongTensor of shape :math:`(E)`, where
            :math:`E` is the number of triples.

        Returns
        -------
        torch.Tensor
            The triple scores. The tensor is of shape :math:`(E)`.
        """
        h_rel = self.rel_emb(rels)

        return - torch.norm(h_head + h_rel - h_tail, p=self.p, dim=-1)

class TransR(nn.Module):
    r"""

    Description
    -----------
    """
    def __init__(self, num_rels, rfeats, nfeats, p=1):
        super(TransR, self).__init__()

        self.rel_emb = nn.Embedding(num_rels, rfeats)
        self.rel_project = nn.Embedding(num_rels, nfeats * rfeats)
        self.rfeats = rfeats
        self.nfeats = nfeats
        self.p = p

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """
        self.rel_emb.reset_parameters()
        self.rel_project.reset_parameters()

    def forward(self, h_head, h_tail, rels):
        r"""

        Description
        -----------
        Score triples.

        Parameters
        ----------
        h_head : torch.Tensor
            Head entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        h_tail : torch.Tensor
            Tail entity features. The tensor is of shape :math:`(E, D)`, where
            :math:`E` is the number of triples, and :math:`D` is the feature size.
        rels : torch.Tensor
            Relation types. It is a LongTensor of shape :math:`(E)`, where
            :math:`E` is the number of triples.

        Returns
        -------
        torch.Tensor
            The triple scores. The tensor is of shape :math:`(E)`.
        """
        h_rel = self.rel_emb(rels)
        proj_rel = self.rel_project(rels).reshape(-1, self.nfeats, self.rfeats)
        h_head = torch.einsum('ab,abc->ac', h_head, proj_rel)
        h_tail = torch.einsum('ab,abc->ac', h_tail, proj_rel)

        return - torch.norm(h_head + h_rel - h_tail, p=self.p, dim=-1)
