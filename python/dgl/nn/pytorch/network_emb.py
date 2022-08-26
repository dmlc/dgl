import torch
from torch import nn, Tensor
import numpy as np
from ...sampling import random_walk
from torch.nn import init
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tqdm


class Metapath2vec(nn.Module):
    """
    Metapath2vec, a scalable representation learning model for heterogeneous networks from KDD’17, August
<https://dl.acm.org/doi/pdf/10.1145/3097983.3098036>
    Parameters
    ----------
    g :DGLHeteroGraph
        Graph data for generating embedding
    emb_dimension :int
        Size of each embedding vector
    metapath :List[str]
        The metapath is described as a list of strings for relation edges between different types of nodes. It is a  sequence of edges which denotes transition  for different types of nodes at each step. Note that the node type at the start and end of the metapath are commonly the same.
    window_size :int
        The context size which is considered for positive samples.
    min_count :int , optional
        The number of node counts that occur less than  min_count  value in all meta-paths will be discarded before random walks. (default: 0)
    negative_samples_size :int , optional
        The number of negative samples need to be extracted for each positive sample.(default: 5)
    node_repeat  :int , optional
        The number of random walks to sample for each start node$V_1$. (default: 1)
    nid2word  :Dict[ str, Dict[ int , str ] ], optional
        If set, we can use model.id2word  to get the dict where the key is embedding id and the value is its corresponding name.(default: None)
    sparse  :int , optional
        If set to True, gradients w.r.t. to the weight matrix will be sparse. (default: True)
    """

    def __init__(self,
                 g,
                 emb_dimension,
                 metapath,
                 window_size,
                 min_count=0,
                 negative_samples_size=5,
                 node_repeat=1,
                 nid2word=None,
                 sparse=True
                 ):
        super(Metapath2vec, self).__init__()
        self.g = g
        self.emb_dimension = emb_dimension
        self.metapath = metapath
        self.min_count = min_count
        self.window_size = window_size
        self.negative_samples_size = negative_samples_size
        self.nid2word = nid2word
        self.sparse = sparse
        self.node_repeat = node_repeat
        self.token_count = 0
        self.walk_dataset = []
        self.negatives = []
        self.discards = dict()
        self.negpos = 0
        self.NEGATIVE_TABLE_SIZE = 1e8
        self._extract()
        self._initTableNegatives()

    def _extract(self):

        assert len(self.metapath) + 1 >= self.window_size
        word_frequency = dict()
        word_frequency_filtered = dict()

        ##build whole vocab for all nodes
        self.token_count = 0
        self.type2id = dict()
        self.id2word = dict()
        wid = 0
        if self.nid2word == None:
            for type in self.g.ntypes:
                self.type2id[type] = dict()
                for idx in range(self.g.num_nodes(type)):
                    self.type2id[type][idx] = wid
                    wid += 1
        else:
            for type in self.g.ntypes:
                self.type2id[type] = dict()
                for idx in range(self.g.num_nodes(type)):
                    self.type2id[type][idx] = wid
                    self.id2word[wid] = self.nid2word[type][idx]
                    wid += 1

        ## convert the edge metapath to the node metapath
        edge2des = {}
        edge2src = {}
        for comb in self.g.canonical_etypes:
            edge2des[comb[1]] = comb[2]
            edge2src[comb[1]] = comb[0]
        nodespath = []
        nodespath.append(edge2src[self.metapath[0]])
        for edge in self.metapath:
            nodespath.append(edge2des[edge])

        ##start random walk
        for idx in tqdm.trange(self.g.number_of_nodes(nodespath[0])):
            traces = random_walk(g=self.g, nodes=[idx, ] * self.node_repeat, metapath=self.metapath)
            for tr in traces[0].numpy():
                line = [self.type2id[nodespath[i]][tr[i]] for i in range(len(tr))]
                self.walk_dataset.append(line)
                if len(line) > 1:
                    for word in line:
                        self.token_count += 1
                        word_frequency[word] = word_frequency.get(word, 0) + 1

        # Filter out the number of words less than the minimum frequency
        t = 0.0001
        for word, freq in word_frequency.items():
            if freq > self.min_count:
                word_frequency_filtered[word] = freq
                f = np.array(freq) / self.token_count
                self.discards[word] = np.sqrt(t / f) + (t / f)

        # get the number of all nodes
        self.word_count = 0
        self.word_frequency = word_frequency_filtered
        for k in self.type2id.keys():
            self.word_count += len(self.type2id[k])

        print("Total embeddings: " + str(self.word_count))
        print("real embeddings：" + str(len(self.word_frequency)))

        self.u_embeddings = nn.Embedding(self.word_count, self.emb_dimension, sparse=self.sparse)
        self.v_embeddings = nn.Embedding(self.word_count, self.emb_dimension, sparse=self.sparse)

    def _initTableNegatives(self):
        # get a table for negative sampling, if word with index 2 appears twice, then 2 will be listed
        # in the table twice.
        pow_frequency = np.array(list(self.word_frequency.values())) ** 0.75
        words_pow = sum(pow_frequency)
        ratio = pow_frequency / words_pow
        count = np.round(ratio * self.NEGATIVE_TABLE_SIZE)
        word_list = list(self.word_frequency.keys())
        for wid, c in enumerate(count):
            self.negatives += [word_list[wid]] * int(c)
        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)
        self.sampling_prob = ratio

    def initParameters(self):
        """
        Initialize the embedding parameters
        """
        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)
        init.constant_(self.v_embeddings.weight.data, 0)

    def _getNegatives(self, target, size):
        response = self.negatives[self.negpos:self.negpos + size]
        self.negpos = (self.negpos + size) % len(self.negatives)
        if len(response) != size:
            return np.concatenate((response, self.negatives[0:self.negpos]))
        return response

    def _generate_sample(self, batches):
        pair_catch = []
        for batch in batches:
            if len(batch) < 0:
                continue
            word_ids = [w for w in batch if w in self.word_frequency and np.random.rand() < self.discards[w]]

            for i, u in enumerate(word_ids):
                for j, v in enumerate(
                        word_ids[max(i - self.window_size, 0):i + self.window_size]):
                    assert u < self.word_count
                    assert v < self.word_count
                    if i == j:
                        continue
                    pair_catch.append((u, v, self._getNegatives(v, self.negative_samples_size)))

        all_u = [u for u, _, _ in pair_catch]
        all_v = [v for _, v, _ in pair_catch]
        all_neg_v = [neg_v for _, _, neg_v in pair_catch]

        return torch.LongTensor(all_u), torch.LongTensor(all_v), torch.LongTensor(all_neg_v)

    def loader(self, **kwargs):
        r"""Returns the data loader that contains center node ，positive context node，and negative samples on the heterogeneous graph random walk.
        Args:
            **kwargs (optional): Arguments of
                :class:`torch.utils.data.DataLoader`, such as
                :obj:`batch_size`, :obj:`shuffle`, :obj:`drop_last` or
                :obj:`num_workers`.
        """
        return DataLoader(self.walk_dataset,
                          collate_fn=self._generate_sample, **kwargs)

    def forward(self, pos_u, pos_v, neg_v):
        """
        Return the loss score given center node,positive context node,and negative samples.
        """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)
        emb_neg_v = self.v_embeddings(neg_v)

        score = torch.sum(torch.mul(emb_u, emb_v), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(emb_neg_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)