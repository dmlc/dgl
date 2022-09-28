"""Network Embedding NN Modules"""
# pylint: disable= invalid-name
import random
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
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

    >>> from dgl.data import CoraGraphDataset
    >>> from torch.utils.data import DataLoader
    >>> from dgl.nn import DeepWalk

    >>> dataset = CoraGraphDataset()
    >>> g = dataset[0]
    >>> model = DeepWalk(g)
    >>> dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
    ...                         shuffle=True, collate_fn=model.sample)
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
        indices : Tensor
            Nodes from which we perform random walk

        Returns
        -------
        Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(len(indices), walk_length)`.
        """
        return random_walk(self.g, indices, length=self.walk_length - 1)[0]

    def forward(self, batch_walk):
        """Compute the loss for the batch of random walks

        Parameters
        ----------
        batch_walk : Tensor
            Random walks in the form of node ID sequences. The Tensor
            is of shape :attr:`(batch_size, walk_length)`.
        """
        batch_size = len(batch_walk)
        device = batch_walk.device

        batch_node_embed = self.node_embed(batch_walk).view(-1, self.emb_dim)
        batch_context_embed = self.context_embed(batch_walk).view(-1, self.emb_dim)

        batch_idx_list_offset = torch.arange(batch_size) * self.walk_length
        batch_idx_list_offset = batch_idx_list_offset.to(device).unsqueeze(1)
        batch_idx_list_src = batch_idx_list_offset + self.idx_list_src.unsqueeze(0)
        batch_idx_list_dst = batch_idx_list_offset + self.idx_list_dst.unsqueeze(0)
        batch_idx_list_src = batch_idx_list_src.view(-1)
        batch_idx_list_dst = batch_idx_list_dst.view(-1)

        pos_src_emb = batch_node_embed[batch_idx_list_src]
        pos_dst_emb = batch_context_embed[batch_idx_list_dst]

    def _generate_sample(self, batch_data):
        if flag:
            self.index_emb_negu, self.index_emb_negv = self._init_emb2neg_index(
                self.walk_length,
                self.window_size,
                self.negative_size,
                self.batch_size,
                device)

        ## Negative
        emb_neg_u = torch.index_select(emb_u, 0, self.index_emb_negu)
        if self.fast_neg:
            emb_neg_v = torch.index_select(emb_v, 0, self.index_emb_negv)
        else:
            neg_nodes = torch.LongTensor(
                np.random.choice(self.neg_table,
                                 self.batch_size * self.num_pos * self.negative_size,
                                 replace=True))
            emb_neg_v = self.context_embed(neg_nodes)

        return emb_pos_u, emb_pos_v, emb_neg_u, emb_neg_v

    def forward(self, emb_pos_u, emb_pos_v, emb_neg_u, emb_neg_v):
        """
        Return the loss score given center nodes, positive context nodes, negative samples
        and corresponding context nodes.

        Parameters
        ----------
        emb_pos_u : Tensor
            center nodes embedding id
        emb_pos_v : Tensor
            positive context nodes embedding id
        emb_neg_u : Tensor
            negative samples embedding id
        emb_neg_v : Tensor
            context nodes corresponding to negative samples

        Returns
        ------
        torch.Tensor
        return the SkipGram model loss.
        """
        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v),dim=1)
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        pos_score = torch.mean(-F.logsigmoid(pos_score))

        # [batch_size * walk_length * negative, dim]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v),dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        # [batch_size * walk_length * negative, 1]
        neg_score = torch.mean(-F.logsigmoid(-neg_score)) * self.negative_size * self.neg_weight

        loss = torch.mean(pos_score + neg_score)
        return loss

    def _init_emb2neg_index(self, walk_length, window_size, negative, batch_size, device):
        '''select embedding of negative nodes from a batch of node embeddings
        for fast negative sampling
        '''
        idx_list_u = []
        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(i - window_size, i):
                    if j >= 0:
                        idx_list_u += [i + b * walk_length] * negative
                for j in range(i + 1, i + 1 + window_size):
                    if j < walk_length:
                        idx_list_u += [i + b * walk_length] * negative

        idx_list_v = list(range(batch_size * walk_length)) \
                     * negative * window_size * 2
        random.shuffle(idx_list_v)
        idx_list_v = idx_list_v[:len(idx_list_u)]

        # [bs * walk_length * negative]
        index_emb_negu = torch.LongTensor(idx_list_u).to(device)
        index_emb_negv = torch.LongTensor(idx_list_v).to(device)

        return index_emb_negu, index_emb_negv
