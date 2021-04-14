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
from .utils import _get_dgl_url, save_graphs, save_info, load_graphs, \
    load_info, deprecate_property
from ..convert import from_networkx

__all__ = ['SST', 'SSTDataset']


class SSTDataset(DGLBuiltinDataset):
    r"""Stanford Sentiment Treebank dataset.

    .. deprecated:: 0.5.0
        
        - ``trees`` is deprecated, it is replaced by:

            >>> dataset = SSTDataset()
            >>> for tree in dataset:
            ....    # your code here

        - ``num_vocabs`` is deprecated, it is replaced by ``vocab_size``.

    Each sample is the constituency tree of a sentence. The leaf nodes
    represent words. The word is a int value stored in the ``x`` feature field.
    The non-leaf node has a special value ``PAD_WORD`` in the ``x`` field.
    Each node also has a sentiment annotation: 5 classes (very negative,
    negative, neutral, positive and very positive). The sentiment label is a
    int value stored in the ``y`` feature field.
    Official site: `<http://nlp.stanford.edu/sentiment/index.html>`_

    Statistics:

    - Train examples: 8,544
    - Dev examples: 1,101
    - Test examples: 2,210
    - Number of classes for each node: 5

    Parameters
    ----------
    mode : str, optional
        Should be one of ['train', 'dev', 'test', 'tiny']
        Default: train
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

    Attributes
    ----------
    vocab : OrderedDict
        Vocabulary of the dataset
    trees : list
        A list of DGLGraph objects
    num_classes : int
        Number of classes for each node
    pretrained_emb: Tensor
        Pretrained glove embedding with respect the vocabulary.
    vocab_size : int
        The size of the vocabulary
    num_vocabs : int
        The size of the vocabulary

    Notes
    -----
    All the samples will be loaded and preprocessed in the memory first.

    Examples
    --------
    >>> # get dataset
    >>> train_data = SSTDataset()
    >>> dev_data = SSTDataset(mode='dev')
    >>> test_data = SSTDataset(mode='test')
    >>> tiny_data = SSTDataset(mode='tiny')
    >>>
    >>> len(train_data)
    8544
    >>> train_data.num_classes
    5
    >>> glove_embed = train_data.pretrained_emb
    >>> train_data.vocab_size
    19536
    >>> train_data[0]
    Graph(num_nodes=71, num_edges=70,
      ndata_schemes={'x': Scheme(shape=(), dtype=torch.int64), 'y': Scheme(shape=(), dtype=torch.int64), 'mask': Scheme(shape=(), dtype=torch.int64)}
      edata_schemes={})
    >>> for tree in train_data:
    ...     input_ids = tree.ndata['x']
    ...     labels = tree.ndata['y']
    ...     mask = tree.ndata['mask']
    ...     # your code here
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
        assert mode in ['train', 'dev', 'test', 'tiny']
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
                    g.add_node(cid, x=SSTDataset.PAD_WORD, y=int(child.label()), mask=0)
                    _rec_build(cid, child)
                g.add_edge(cid, nid)

        # add root
        g.add_node(0, x=SSTDataset.PAD_WORD, y=int(root.label()), mask=0)
        _rec_build(0, root)
        ret = from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def has_cache(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        return os.path.exists(graph_path) and os.path.exists(vocab_path)

    def save(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        save_graphs(graph_path, self._trees)
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        save_info(vocab_path, {'vocab': self.vocab})
        if self.pretrained_emb:
            emb_path = os.path.join(self.save_path, 'emb.pkl')
            save_info(emb_path, {'embed': self.pretrained_emb})

    def load(self):
        graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
        vocab_path = os.path.join(self.save_path, 'vocab.pkl')
        emb_path = os.path.join(self.save_path, 'emb.pkl')

        self._trees = load_graphs(graph_path)[0]
        self._vocab = load_info(vocab_path)['vocab']
        self._pretrained_emb = None
        if os.path.exists(emb_path):
            self._pretrained_emb = load_info(emb_path)['embed']

    @property
    def trees(self):
        deprecate_property('dataset.trees', '[dataset[i] for i in len(dataset)]')
        return self._trees

    @property
    def vocab(self):
        r""" Vocabulary

        Returns
        -------
        OrderedDict
        """
        return self._vocab

    @property
    def pretrained_emb(self):
        r"""Pre-trained word embedding, if given."""
        return self._pretrained_emb

    def __getitem__(self, idx):
        r""" Get graph by index

        Parameters
        ----------
        idx : int

        Returns
        -------
        :class:`dgl.DGLGraph`

            graph structure, word id for each node, node labels and masks.

            - ``ndata['x']``: word id of the node
            - ``ndata['y']:`` label of the node
            - ``ndata['mask']``: 1 if the node is a leaf, otherwise 0
        """
        return self._trees[idx]

    def __len__(self):
        r"""Number of graphs in the dataset."""
        return len(self._trees)

    @property
    def num_vocabs(self):
        deprecate_property('dataset.num_vocabs', 'dataset.vocab_size')
        return self.vocab_size

    @property
    def vocab_size(self):
        r"""Vocabulary size."""
        return len(self._vocab)

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 5


SST = SSTDataset
