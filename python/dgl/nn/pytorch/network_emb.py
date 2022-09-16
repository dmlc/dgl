"""Network Embedding NN modules"""
# pylint: disable= invalid-name
import time
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
    <https://arxiv.org/pdf/1403.6652.pdf>`__

    It consists of two main components, a random walk generator, and an update procedure.
    The random walk generator takes a graph :math:`G` and samples uniformly a random vertex
    :math:`v_i` as the root of the random walk :math:`\mathcal{W}_{vi}`. A walk samples uniformly
    from the neighbors of the last vertex visited until the maximum length (t) is reached.
    In practice, We specify a number of random walks γ of length t to start at each vertex. For
    each random walk, We use the SkipGram algorithm and negative sampling to update these
    representations .

    For each node in random walk, which can be regarded as the center node, we can sample positive
    node within context size and draw some negative samples among all node instances from all the
    random walks. Then we can center-context paired nodes and context-negative paired nodes to
    update the network. The formula is as follows:

    .. math::
        log \sigma(X_{c_t}·X_v) + \sum_{m=1}^M Eu^m \sim P(u)[log \sigma(-X_{u_m}·X_v)] ,
        where \sigma(x) = \frac{1}{1+e^{-x}}

    Parameters
    ----------
    g : DGLGraph
        Graph data for generating embedding.
    emb_dim : int
        Size of each embedding vector
    walk_length : int
        The number of nodes in a random walk sequence.
    context_size : int
        The context size which is considered for positive samples.
    num_random_walk : int , optional
        The number of random walks to sample for each start node. Default: 1
    neg_weight : float, optional
        Weight of negative samples in loss function. Default: 1.0
    use_context_weight : bool, optional
        Give different weights to the nodes within the context size. Default: True
    num_procs : int, optional
        The num of split data for mutiprocessing
    negative_size : int , optional
        The number of negative samples to use for each positive sample. Default: 5
    fast_neg : bool, optional
        If set to True, we do negative sampling inside a batch. Default: True
    sparse : bool, optional
        If set to True, gradients with respect to the learnable weights will be sparse.
        Default: True

    """

    def __init__(self,
                 g,
                 emb_dim,
                 walk_length,
                 context_size,
                 num_random_walk,
                 neg_weight=1,
                 use_context_weight=False,
                 num_procs=1,
                 negative_size=5,
                 fast_neg=True,
                 sparse=True
                 ):
        super().__init__()
        self.g = g
        self.emb_dim = emb_dim
        self.context_size = context_size
        self.walk_length = walk_length
        self.neg_weight = neg_weight
        self.use_context_weight = use_context_weight
        self.negative_size = negative_size
        self.batch_siz = 0
        self.num_nodes = self.g.number_of_nodes()
        self.fast_neg = fast_neg

        # content embedding
        self.u_embeddings = nn.Embedding(
            self.num_nodes, self.emb_dim, sparse=sparse)
        # context embedding
        self.v_embeddings = nn.Embedding(
            self.num_nodes, self.emb_dim, sparse=sparse)

        # random walk seeds
        start = time.time()
        valid_seeds = self.g.out_degrees().nonzero().squeeze(-1)
        if len(valid_seeds) != self.num_nodes:
            print("WARNING: The node ids are not serial. Some nodes are invalid.")

        seeds = torch.cat([torch.LongTensor(valid_seeds)] * num_random_walk)
        index = torch.torch.randperm(seeds.size()[0])
        shuffled_seeds = seeds[index]
        self.seeds = torch.split(shuffled_seeds,
                                 int(np.ceil(len(valid_seeds) * num_random_walk / num_procs)),
                                 0)
        end = time.time()
        t = end - start
        print("%d seeds in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        if not fast_neg:
            node_degree = self.g.out_degrees(valid_seeds).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=np.int)
            self.neg_table = []

            for idx, node in enumerate(valid_seeds):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.long)
            del node_degree

        # number of positive node pairs in a sequence
        self.num_pos = int(2 * self.walk_length * self.context_size\
            - self.context_size * (self.context_size + 1))

    def generate_walk(self, i=0):
        '''
        generate random walk on the split data slice i.

        Parameters
        ----------
        i : int, optional
            The split data slice i generated by attr:`num_procs`

        Returns
        -------
        torch.utils.data.DataLoader (Tensor,Tensor,Tensor)
        return the data loader that yields batched data: center node, positive context node,
            and negative samples
        '''
        walks = random_walk(self.g, self.seeds[i], length=self.walk_length - 1)[0]
        return walks

    def loader(self, walks, **kwargs):
        r"""Returns the data loader that yields center node, positive context node,
            negative samples and corresponding context nodes on the random walk.

        Parameters
        ----------
        kwargs : dict, optional
           Key-word arguments to be passed to the parent PyTorch
           :py:class:`torch.utils.data.DataLoader` class. Common arguments are:

             - ``batch_size`` (int): The number of indices in each batch.
             - ``drop_last`` (bool): Whether to drop the last incomplete batch.
             - ``shuffle`` (bool): Whether to randomly shuffle the indices at each epoch.

        Returns
        -------
        torch.Tensor
        return random walks on split data slice.
        """
        return DataLoader(walks,
                          collate_fn=self._generate_sample, **kwargs)

    def _generate_sample(self, batch_data):

        flag = self.batch_size != len(batch_data)
        self.batch_size = len(batch_data)
        device = batch_data[0].device

        if flag:
            # indexes to select positive/negative node pairs from batch_walks
            self.index_emb_posu, self.index_emb_posv = self._init_emb2pos_index(
                self.walk_length,
                self.context_size,
                self.batch_size,
                device)
            self.index_emb_negu, self.index_emb_negv = self._init_emb2neg_index(
                self.walk_length,
                self.context_size,
                self.negative_size,
                self.batch_size,
                device)
            if self.use_context_weight:
                self.context_weight = self._init_weight(
                    self.walk_length,
                    self.context_size,
                    self.batch_size,
                    device)

        batch_data = torch.stack(batch_data)
        emb_u = self.u_embeddings(batch_data).view(-1, self.emb_dim)
        emb_v = self.v_embeddings(batch_data).view(-1, self.emb_dim)

        ## Postive
        # num_pos: the number of positive node pairs generated by a single walk sequence
        # [batch_size * num_pos, dim]
        emb_pos_u = torch.index_select(emb_u, 0, self.index_emb_posu)
        emb_pos_v = torch.index_select(emb_v, 0, self.index_emb_posv)

        ## Negative
        emb_neg_u = torch.index_select(emb_u, 0, self.index_emb_negu)
        if self.fast_neg:
            emb_neg_v = torch.index_select(emb_v, 0, self.index_emb_negv)
        else:
            neg_nodes = torch.LongTensor(
                np.random.choice(self.neg_table,
                                 self.batch_size * self.num_pos * self.negative_size,
                                 replace=True))
            emb_neg_v = self.v_embeddings(neg_nodes)

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
        pos_score = torch.sum(torch.mul(emb_pos_u, emb_pos_v),dim=1) * self.context_weight
        pos_score = torch.clamp(pos_score, max=6, min=-6)
        pos_score = torch.mean(-F.logsigmoid(pos_score))

        # [batch_size * walk_length * negative, dim]
        neg_score = torch.sum(torch.mul(emb_neg_u, emb_neg_v),dim=1)
        neg_score = torch.clamp(neg_score, max=6, min=-6)
        # [batch_size * walk_length * negative, 1]
        neg_score = torch.mean(-F.logsigmoid(-neg_score)) * self.negative_size * self.neg_weight

        loss = torch.mean(pos_score + neg_score)
        return loss


    def _init_emb2pos_index(self, walk_length, context_size, batch_size, device):
        ''' select embedding of positive nodes from a batch of node embeddings
        '''
        idx_list_u = []
        idx_list_v = []
        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(i - context_size, i):
                    if j >= 0:
                        idx_list_u.append(j + b * walk_length)
                        idx_list_v.append(i + b * walk_length)
                for j in range(i + 1, i + 1 + context_size):
                    if j < walk_length:
                        idx_list_u.append(j + b * walk_length)
                        idx_list_v.append(i + b * walk_length)

        # [num_pos * batch_size]
        index_emb_posu = torch.LongTensor(idx_list_u).to(device)
        index_emb_posv = torch.LongTensor(idx_list_v).to(device)

        return index_emb_posu, index_emb_posv

    def _init_emb2neg_index(self, walk_length, context_size, negative, batch_size, device):
        '''select embedding of negative nodes from a batch of node embeddings
        for fast negative sampling
        '''
        idx_list_u = []
        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(i - context_size, i):
                    if j >= 0:
                        idx_list_u += [i + b * walk_length] * negative
                for j in range(i + 1, i + 1 + context_size):
                    if j < walk_length:
                        idx_list_u += [i + b * walk_length] * negative

        idx_list_v = list(range(batch_size * walk_length)) \
                     * negative * context_size * 2
        random.shuffle(idx_list_v)
        idx_list_v = idx_list_v[:len(idx_list_u)]

        # [bs * walk_length * negative]
        index_emb_negu = torch.LongTensor(idx_list_u).to(device)
        index_emb_negv = torch.LongTensor(idx_list_v).to(device)

        return index_emb_negu, index_emb_negv


    def _init_weight(self, walk_length, context_size, batch_size, device):
        ''' init context weight '''
        weight = []
        for b in range(batch_size):
            for i in range(walk_length):
                for j in range(i-context_size, i):
                    if j >= 0:
                        weight.append(1. - float(i - j - 1)/float(context_size))
                for j in range(i + 1, i + 1 + context_size):
                    if j < walk_length:
                        weight.append(1. - float(j - i - 1)/float(context_size))

        # [num_pos * batch_size]
        return torch.Tensor(weight).unsqueeze(1).to(device)

    def reset_parameters(self):
        """
        Initialize the embedding parameters
        """
        initrange = 1.0 / self.emb_dim
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)