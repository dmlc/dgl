"""Network Embedding NN Modules"""
# pylint: disable= invalid-name
import random
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F

from ...random import choice
from ...sampling import random_walk

__all__ = ['DeepWalk']

class DeepWalk(nn.Module):
    """DeepWalk module from `DeepWalk: Online Learning of Social Representations
    <https://arxiv.org/abs/1403.6652>`__

    For a graph, it learns the node representations from scratch by maximizing the similarity of
    node pairs that are nearby (positive node pairs) and minimizing the similarity of other
    random node pairs (negative node pairs).

    Parameters
    ----------
    g : DGLGraph
        Graph for learning node embeddings
    emb_dim : int, optional
        Size of each embedding vector. Default: 128
    walk_length : int, optional
        Number of nodes in a random walk sequence. Default: 40
    window_size : int, optional
        In a random walk :attr:`w`, a node :attr:`w[j]` is considered close to a node
        :attr:`w[i]` if :attr:`i - window_size <= j <= i + window_size`. Default: 5
    neg_weight : float, optional
        Weight of the loss term for negative samples in the total loss. Default: 1.0
    negative_size : int, optional
        Number of negative samples to use for each positive sample in an iteration. Default: 1
    fast_neg : bool, optional
        If True, it samples negative node pairs within a batch of random walks. Default: True
    sparse : bool, optional
        If True, gradients with respect to the learnable weights will be sparse.
        Default: True

    Attributes
    ----------
    node_embed : nn.Embedding
        Embedding table of the nodes

    Examples
    --------

    >>> import torch
    >>> from dgl.data import CoraGraphDataset
    >>> from dgl.nn import DeepWalk
    >>> from torch.optim import SparseAdam
    >>> from torch.utils.data import DataLoader
    >>> from sklearn.linear_model import LogisticRegression

    >>> dataset = CoraGraphDataset()
    >>> g = dataset[0]
    >>> model = DeepWalk(g)
    >>> optimizer = SparseAdam(model.parameters(), lr=0.01)
    >>> dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
    ...                         shuffle=True, collate_fn=model.sample)
    >>> num_epochs = 5
    >>> for epoch in range(num_epochs):
    ...     for batch_walk in dataloader:
    ...         loss = model(batch_walk)
    ...         loss.backward()
    ...         optimizer.step()
    >>> train_mask = g.ndata['train_mask']
    >>> test_mask = g.ndata['test_mask']
    >>> X = model.node_embed.weight.detach()
    >>> y = g.ndata['label']
    >>> clf = LogisticRegression().fit(X[train_mask].numpy(), y[train_mask].numpy())
    >>> clf.score(X[test_mask].numpy(), y[test_mask].numpy())
    """
    def __init__(self,
                 g,
                 emb_dim=128,
                 walk_length=40,
                 window_size=5,
                 neg_weight=1,
                 negative_size=5,
                 fast_neg=True,
                 sparse=True):
        super().__init__()

        assert walk_length >= window_size + 1, \
            f'Expect walk_length >= window_size + 1, got {walk_length} and {window_size + 1}'

        self.g = g
        self.emb_dim = emb_dim
        self.window_size = window_size
        self.walk_length = walk_length
        self.neg_weight = neg_weight
        self.negative_size = negative_size
        self.fast_neg = fast_neg

        num_nodes = g.num_nodes()

        # center node embedding
        self.node_embed = nn.Embedding(num_nodes, emb_dim, sparse=sparse)
        # context embedding
        self.context_embed = nn.Embedding(num_nodes, emb_dim, sparse=sparse)
        self.reset_parameters()

        if not fast_neg:
            node_degree = g.out_degrees().pow(0.75)
            # categorical distribution for true negative sampling
            self.neg_prob = node_degree / node_degree.sum()

        # Get list index pairs for positive samples.
        # Given i, positive index pairs are (i - window_size, i), ... ,
        # (i - 1, i), (i + 1, i), ..., (i + window_size, i)
        idx_list_src = []
        idx_list_dst = []

        for i in range(walk_length):
            for j in range(max(0, i - window_size), i):
                idx_list_src.append(j)
                idx_list_dst.append(i)
            for j in range(i + 1, min(walk_length, i + 1 + window_size)):
                idx_list_src.append(j)
                idx_list_dst.append(i)

        self.idx_list_src = torch.LongTensor(idx_list_src)
        self.idx_list_dst = torch.LongTensor(idx_list_dst)

    def reset_parameters(self):
        """Reinitialize learnable parameters"""
        init_range = 1.0 / self.emb_dim
        init.uniform_(self.node_embed.weight.data, -init_range, init_range)
        init.constant_(self.context_embed.weight.data, 0)

    def sample(self, indices):
        """Sample random walks

        Parameters
        ----------
        indices : torch.Tensor
            Nodes from which we perform random walk

        Returns
        -------
        torch.Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(len(indices), walk_length)`.
        """
        return random_walk(self.g, indices, length=self.walk_length - 1)[0]

    def forward(self, batch_walk):
        """Compute the loss for the batch of random walks

        Parameters
        ----------
        batch_walk : torch.Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(batch_size, walk_length)`.

        Returns
        -------
        torch.Tensor
            Loss value
        """
        batch_size = len(batch_walk)
        device = batch_walk.device

        batch_node_embed = self.node_embed(batch_walk).view(-1, self.emb_dim)
        batch_context_embed = self.context_embed(batch_walk).view(-1, self.emb_dim)

        batch_idx_list_offset = torch.arange(batch_size) * self.walk_length
        batch_idx_list_offset = batch_idx_list_offset.unsqueeze(1)
        idx_list_src = batch_idx_list_offset + self.idx_list_src.unsqueeze(0)
        idx_list_dst = batch_idx_list_offset + self.idx_list_dst.unsqueeze(0)
        idx_list_src = idx_list_src.view(-1).to(device)
        idx_list_dst = idx_list_dst.view(-1).to(device)

        pos_src_emb = batch_node_embed[idx_list_src]
        pos_dst_emb = batch_context_embed[idx_list_dst]

        neg_idx_list_src = idx_list_dst.unsqueeze(1) + torch.zeros(
            self.negative_size).unsqueeze(0).to(device)
        neg_idx_list_src = neg_idx_list_src.view(-1)
        neg_src_emb = batch_node_embed[neg_idx_list_src.long()]

        if self.fast_neg:
            neg_idx_list_dst = list(range(batch_size * self.walk_length)) \
                * (self.negative_size * self.window_size * 2)
            random.shuffle(neg_idx_list_dst)
            neg_idx_list_dst = neg_idx_list_dst[:len(neg_idx_list_src)]
            neg_idx_list_dst = torch.LongTensor(neg_idx_list_dst).to(device)
            neg_dst_emb = batch_context_embed[neg_idx_list_dst]
        else:
            neg_dst = choice(self.g.num_nodes(), size=len(neg_src_emb), prob=self.neg_prob)
            neg_dst_emb = self.context_embed(neg_dst.to(device))

        pos_score = torch.sum(torch.mul(pos_src_emb, pos_dst_emb), dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        pos_score = torch.mean(-F.logsigmoid(pos_score))

        neg_score = torch.sum(torch.mul(neg_src_emb, neg_dst_emb), dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        neg_score = torch.mean(-F.logsigmoid(-neg_score)) * self.negative_size * self.neg_weight

        return torch.mean(pos_score + neg_score)
