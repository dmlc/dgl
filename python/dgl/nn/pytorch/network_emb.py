"""Network Embedding NN modules"""
# pylint: disable= invalid-name
import torch
from torch import nn
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import tqdm
from collections import defaultdict
from ...sampling import random_walk

__all__ = ['MetaPath2Vec']

class MetaPath2Vec(nn.Module):
    """metapath2vec module from `metapath2vec: Scalable Representation Learning for
    Heterogeneous Networks <https://dl.acm.org/doi/pdf/10.1145/3097983.3098036>`__

    To achieve efficient optimization, we leverage the negative sampling technique for the
    training process. For each node in meta-path, which can be regarded as the center node,
    we can sample positive node within context size and draw some negative samples among all
    types of nodes from all meta-paths. Then we can use the center-context paired nodes and
    context-negative paired nodes to update the network. The formula is as follows:

    .. math::
        log \sigma(X_{c_t}·X_v) + \sum_{m=1}^M Eu^m \sim P(u)[log \sigma(-X_{u_m}·X_v)] ,
        where \sigma(x) = \frac{1}{1+e^{-x}}
    Parameters
    ----------
    g : DGLGraph
        Graph data for generating embedding. Note we cannot have the same edge type across
        different canonical edge types.
    emb_dim : int
        Size of each embedding vector
    metapath : List[str]
        A sequence of edge types in the form of string. It defines a new edge type by composing
        multiple edge types in order. Note that the start node type and the end one are commonly
        the same.
    context_size : int
        The context size which is considered for positive samples.
    min_count : int , optional
        The nodes that appear less than :attr:`min_count` times in all meta-path instances will
        be discarded, which means we can not get the their trained embedding at last Default: 0
    negative_size : int , optional
        The number of negative samples to use for each positive sample. Default: 5
    num_random_walk : int , optional
        The number of random walks to sample for each start node. Default: 1
    nid2word : Dict[str, Dict[int , str]], optional
        If set, we can use model.id2word  to get the dict where the key is embedding id and
        the value is its corresponding name. Default: None
    sparse : bool, optional
        If set to True, gradients with respect to the learnable weights will be sparse.
        Default: True
    negative_table_size : float, optional
        This is used with the node frequency to build the table for negative sampling
    fre_subsampling : bool, optional
        If set, a subsampling approach is applied to adjust node frequency

    Examples
    --------
    >>> import torch
    >>> from torch import optim
    >>> import dgl
    >>> from dgl.nn.pytorch import MetaPath2Vec

    >>> # Define a model
    >>> g = dgl.heterograph({
...         ('user', 'uc', 'company'): ([0, 0, 2, 1, 3], [1, 2, 1, 3, 0]),
...         ('company', 'cp', 'product'): ([0, 0, 0, 1, 2, 3], [0, 2, 3, 0, 2, 1]),
...         ('company', 'cu', 'user'): ([1, 2, 1, 3, 0], [0, 0, 2, 1, 3]),
...         ('product', 'pc', 'company'): ([0, 2, 3, 0, 2, 1], [0, 0, 0, 1, 2, 3])
...     })
    >>> model = MetaPath2Vec(g, 64, ['uc','cu'], 1, 1, 2, 10, fre_subsampling=False)
    >>> print(model.local_to_global_id)
    {'company': array([0, 1, 2, 3]), 'user': array([4, 5, 6, 7])}

    >>> # Train the model
    >>> dataloader=model.loader()
    >>> optimizer = optim.SparseAdam(list(model.parameters()), lr=0.025)
    >>> scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(dataloader))
    >>> running_loss = 0.0
    >>> for i, sample_batched in enumerate((dataloader)):
...     pos_u = sample_batched[0]
...     pos_v = sample_batched[1]
...     neg_v = sample_batched[2]
...     scheduler.step()
...     optimizer.zero_grad()
...     loss = model(pos_u, pos_v, neg_v)
...     loss.backward()
...     optimizer.step()
...     running_loss = running_loss * 0.9 + loss.item() * 0.1
    """
    def __init__(self,
                 g,
                 emb_dim,
                 metapath,
                 context_size,
                 min_count=0,
                 negative_size=5,
                 num_random_walk=1,
                 nid2word=None,
                 sparse=True,
                 negative_table_size=1e8,
                 fre_subsampling=True
                 ):
        super().__init__()
        self.g = g
        self.emb_dim = emb_dim
        self.metapath = metapath
        self.min_count = min_count
        self.context_size = context_size
        self.negative_size = negative_size
        self.nid2word = nid2word
        self.sparse = sparse
        self.fre_subsampling=fre_subsampling
        self.num_random_walk = num_random_walk
        self.walk_dataset = []
        self.negatives = []
        self.discards = dict()
        self.negpos = 0
        self.negative_table_size = negative_table_size
        self._extract()
        self._init_table_negatives()

    def _extract(self):
        # The metapath node length must be longer than the context size. Otherwise there are not
        # enough nodes to sample.
        assert len(self.metapath) + 1 >= self.context_size
        node_frequency = defaultdict(int)
        node_frequency_filtered = dict()

        ## convert the edge metapath to the node metapath
        nodespath = []
        edge2des = {}
        for src_type, etype, dst_type in self.g.canonical_etypes:
            edge2des[etype] = dst_type
            if etype == self.metapath[0]:
                nodespath.append(src_type)
        for edge in self.metapath:
            nodespath.append(edge2des[edge])

        ##build whole vocab for all nodes
        self.node_count = 0
        self.token_count = 0
        self.local_to_global_id = dict()
        self.id2word = dict()
        if self.nid2word is None:
            for nodetype in set(nodespath):
                self.local_to_global_id[nodetype] = np.arange(self.g.num_nodes(nodetype)) \
                                                    + self.node_count
                self.node_count += self.g.num_nodes(nodetype)
        else:
            wid = 0
            for nodetype in set(nodespath):
                self.local_to_global_id[nodetype] = dict()
                for idx in range(self.g.num_nodes(nodetype)):
                    self.local_to_global_id[nodetype][idx] = wid
                    self.id2word[wid] = self.nid2word[nodetype][idx]
                    wid += 1
            self.node_count=self.g.num_nodes

        # start random walk
        for idx in tqdm.trange(self.g.num_nodes(nodespath[0])):
            traces, _ = random_walk(g=self.g, nodes=[idx] * self.num_random_walk, metapath=self.metapath)
            for tr in traces.cpu().numpy():
                line = [self.local_to_global_id[nodespath[i]][tr[i]] for i in range(len(tr))]
                self.walk_dataset.append(line)
                if len(line) > 1:
                    self.token_count += len(line)
                    for node in line:
                        node_frequency[node] = node_frequency[node] + 1

        # Filter out the nodes whose number of occurrences is smaller than min_count and frequency
        # subsampling
        t = 0.0001 if self.fre_subsampling else 1
        for node, freq in node_frequency.items():
            if freq > self.min_count:
                node_frequency_filtered[node] = freq
                f = np.array(freq) / self.token_count
                self.discards[node] = np.sqrt(t / f) + (t / f)

        self.node_frequency = node_frequency_filtered

        print("Total embeddings: " + str(self.node_count))
        print("Real embeddings: " + str(len(self.node_frequency)))

        self.u_emb = nn.Embedding(self.node_count, self.emb_dim, sparse=self.sparse)
        self.v_emb = nn.Embedding(self.node_count, self.emb_dim, sparse=self.sparse)

    def _init_table_negatives(self):
        # get a table for negative sampling, if node with index 2 appears twice, then 2
        # will be listed in the table twice.
        pow_frequency = np.array(list(self.node_frequency.values())) ** 0.75
        nodes_pow = sum(pow_frequency)
        ratio = pow_frequency / nodes_pow
        count = np.round(ratio * self.negative_table_size)
        node_list = list(self.node_frequency.keys())
        for wid, c in enumerate(count):
            self.negatives += [node_list[wid]] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def reset_parameters(self):
        """
        Initialize the embedding parameters
        """
        init_range = 1.0 / self.emb_dim
        init.uniform_(self.u_emb.weight.data, -init_range, init_range)
        init.constant_(self.v_emb.weight.data, 0)

    def _getNegatives(self):
        # TODO online sampling
        response = self.negatives[self.negpos:self.negpos + self.negative_size]
        self.negpos = (self.negpos + self.negative_size) % len(self.negatives)
        if len(response) != self.negative_size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def _generate_sample(self, batches):
        pair_catch = []
        for batch in batches:
            # TODO check whether we should directly update self.walk_dataset at the beginning
            node_ids = [w for w in batch if w in self.node_frequency
                        and np.random.rand() < self.discards[w]]

            for i, u in enumerate(node_ids):
                for j, v in enumerate(
                        node_ids[max(i - self.context_size, 0):i + self.context_size]):
                    if i == j:
                        continue
                    pair_catch.append((u, v, self._getNegatives()))

        all_u = [u for u, _, _ in pair_catch]
        all_v = [v for _, v, _ in pair_catch]
        all_neg_v = [neg_v for _, _, neg_v in pair_catch]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)

    def loader(self, **kwargs):
        r"""Returns the data loader that yields center node, positive context node,
            and negative samples on the heterogeneous graph random walk.

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
        torch.utils.data.DataLoader (Tensor,Tensor,Tensor)
        return the data loader that yields batched data: center node, positive context node,
            and negative samples
        """
        return DataLoader(self.walk_dataset,
                          collate_fn=self._generate_sample, **kwargs)

    def forward(self, pos_u, pos_v, neg_v):
        """
        Return the loss score given center nodes, positive context nodes, and negative samples.

        Parameters
        ----------
        pos_u : Tensor
            center nodes embedding id
        pos_v : Tensor
            positive context nodes embedding id
        neg_v : Tensor
            negative samples embedding id
        Returns
        ------
        torch.Tensor
        return the SkipGram model loss given center nodes id, positive context nodes id,
        and negative samples id.
        """
        emb_u = self.u_emb(pos_u)
        emb_v = self.v_emb(pos_v)
        emb_neg_v = self.v_emb(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

