"""Tree-structured data.

Including:
    - Stanford Sentiment Treebank
"""
from __future__ import absolute_import

from collections import OrderedDict
import networkx as nx

import numpy as np
import os

from .dgl_dataset import DGLBuiltinDataset
from .. import backend as F
from ..graph import DGLGraph
from .utils import _get_dgl_url, save_graphs, save_info, load_graphs, \
    load_info, deprecate_property, deprecate_class

__all__ = ['SST', 'SSTDataset']


class SSTDataset(DGLBuiltinDataset):
    r"""Stanford Sentiment Treebank dataset.

    Each sample is the constituency tree of a sentence. The leaf nodes
    represent words. The word is a int value stored in the ``x`` feature field.
    The non-leaf node has a special value ``PAD_WORD`` in the ``x`` field.
    Each node also has a sentiment annotation: 5 classes (very negative,
    negative, neutral, positive and very positive). The sentiment label is a
    int value stored in the ``y`` feature field.

    Official site: http://nlp.stanford.edu/sentiment/index.html

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    .. note::
        All the samples will be loaded and preprocessed in the memory first.

    Statistics
    ----------
    Train examples: 8544
    Dev examples: 2
    Test examples: 2
    Number of classes for each node: 5

    Parameters
    ----------
    mode : str, optional
        Can be ``'train'``, ``'dev'``, ``'test'`` and specifies which data file to use.
    glove_embed_file : str, optional
        The path to pretrained glove embedding file.
        Default: None
    vocab_file : str, optional
        Optional vocabulary file. If not given, the default vacabulary file is used.
        Default: None
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Returns
    -------
    SSTDataset object with five properties:
        vocab : OrderedDict, vocabulary of the dataset
        trees : list, a list of DGLGraph objects each contains
            - ndata['x']: word id of the node
            - ndata['y']: label of the node
            - ndata['mask']: 1 if the node is a leaf, otherwise 0
        num_classes : int, number of classes for each node
        pretrained_emb: float tensor, pretrained glove embedding with respect the vocabulary.
        vocab_size : int, the size of the vocabulary

    Examples
    --------
    >>> # get dataset
    >>> train_data = SSTDataset()
    >>> dev_data = SSTDataset(mode='dev')
    >>> test_data = SSTDataset(mode='test')
    >>>
    >>> len(train_data.trees)
    8544
    >>> train_data.num_classes
    5
    >>> glove_embed = train_data.pretrained_emb
    >>> train_data.vocab_size
    19536
    >>> train_data.trees[0]
    DGLGraph(num_nodes=71, num_edges=70,
         ndata_schemes={'x': Scheme(shape=(), dtype=torch.int64), 'y': Scheme(shape=(),
          dtype=torch.int64), 'mask': Scheme(shape=(), dtype=torch.int64)}
         edata_schemes={})
    >>> for tree in train_data:
    ...     input_ids = tree.ndata['x']
    ...     labels = tree.ndata['y']
    ...     mask = tree.ndata['mask']
    ...     # your code here
    >>>
    """

    PAD_WORD = -1  # special pad word id
    UNK_WORD = -1  # out-of-vocabulary word id

    def __init__(self,
                 mode='train',
                 glove_embed_file=None,
                 vocab_file=None,
                 raw_dir=None,
                 force_reload=False,
                 verbose=False):
        assert mode in ['train', 'dev', 'test']
        _url = _get_dgl_url('dataset/sst.zip')
        self._glove_embed_file = glove_embed_file if mode == 'train' else None
        self.mode = mode
        self._vocab_file = vocab_file
        super(SSTDataset, self).__init__(name='sst',
                                         url=_url,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def process(self):
        from nltk.corpus.reader import BracketParseCorpusReader
        # load vocab file
        self._vocab = OrderedDict()
        vocab_file = self._vocab_file if self._vocab_file is not None else os.path.join(self.raw_path, 'vocab.txt')
        with open(vocab_file, encoding='utf-8') as vf:
            for line in vf.readlines():
                line = line.strip()
                self._vocab[line] = len(self._vocab)

        # filter glove
        if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
            glove_emb = {}
            with open(self._glove_embed_file, 'r', encoding='utf-8') as pf:
                for line in pf.readlines():
                    sp = line.split(' ')
                    if sp[0].lower() in self._vocab:
                        glove_emb[sp[0].lower()] = np.asarray([float(x) for x in sp[1:]])
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader(self.raw_path, files)
        sents = corpus.parsed_sents(files[0])

        # initialize with glove
        pretrained_emb = []
        fail_cnt = 0
        for line in self._vocab.keys():
            if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
                if not line.lower() in glove_emb:
                    fail_cnt += 1
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

        self._pretrained_emb = None
        if self._glove_embed_file is not None and os.path.exists(self._glove_embed_file):
            self._pretrained_emb = F.tensor(np.stack(pretrained_emb, 0))
            print('Miss word in GloVe {0:.4f}'.format(1.0 * fail_cnt / len(self._pretrained_emb)))
        # build trees
        self._trees = []
        for sent in sents:
            self._trees.append(self._build_tree(sent))

    def _build_tree(self, root):
        g = nx.DiGraph()

        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str) or isinstance(child[0], bytes):
                    # leaf node
                    word = self.vocab.get(child[0].lower(), self.UNK_WORD)
                    g.add_node(cid, x=word, y=int(child.label()), mask=1)
                else:
                    g.add_node(cid, x=SST.PAD_WORD, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)
        # add root
        g.add_node(0, x=SST.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        ret = os.path.exists(graph_path)
        if self.mode == 'train':
            info_path = os.path.join(self.save_path, 'graph_info.pkl')
            ret = ret and os.path.exists(info_path)
        return ret

    def save(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self.trees)
        if self.mode == 'train':
            info_path = os.path.join(self.save_path, 'info.pkl')
            save_info(info_path, {'vocab': self.vocab, 'embed': self.pretrained_emb})

    def load(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        self._trees = load_graphs(graph_path)
        info_path = os.path.join(self.save_path, 'info.pkl')
        if os.path.exists(info_path):
            info = load_info(info_path)
            self._vocab = info['vocab']
            self._pretrained_emb = info['embed']

    @property
    def trees(self):
        return self._trees

    @property
    def vocab(self):
        return self._vocab

    @property
    def pretrained_emb(self):
        return self._pretrained_emb

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    @property
    def num_vocabs(self):
        deprecate_property('dataset.num_vocabs', 'dataset.vocab_size')
        return len(self.vocab)

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def num_classes(self):
        return 5


class SST(SSTDataset):
    def __init__(self, mode='train', vocab_file=None):
        deprecate_class('SST', 'SSTDataset')
        super(SST, self).__init__(mode=mode, vocab_file=vocab_file)
