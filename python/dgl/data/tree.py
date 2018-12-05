"""Tree-structured data.

Including:
    - Stanford Sentiment Treebank
"""
from __future__ import absolute_import

from collections import namedtuple, OrderedDict
import networkx as nx

import numpy as np
import os
import dgl
import dgl.backend as F
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

__all__ = ['SSTBatch', 'SST']

_urls = {
    'sst' : 'dataset/sst.zip',
}

SSTBatch = namedtuple('SSTBatch', ['graph', 'mask', 'wordid', 'label'])

class SST(object):
    """Stanford Sentiment Treebank dataset.

    Each sample is the constituency tree of a sentence. The leaf nodes
    represent words. The word is a int value stored in the ``x`` feature field.
    The non-leaf node has a special value ``PAD_WORD`` in the ``x`` field.
    Each node also has a sentiment annotation: 5 classes (very negative,
    negative, neutral, positive and very positive). The sentiment label is a
    int value stored in the ``y`` feature field.

    .. note::
        This dataset class is compatible with pytorch's :class:`Dataset` class.

    .. note::
        All the samples will be loaded and preprocessed in the memory first.

    Parameters
    ----------
    mode : str, optional
        Can be ``'train'``, ``'val'``, ``'test'`` and specifies which data file to use.
    vocab_file : str, optional
        Optional vocabulary file.
    """
    PAD_WORD=-1  # special pad word id
    UNK_WORD=-1  # out-of-vocabulary word id
    def __init__(self, mode='train', vocab_file=None):
        self.mode = mode
        self.dir = get_download_dir()
        self.zip_file_path='{}/sst.zip'.format(self.dir)
        self.pretrained_file = 'glove.840B.300d.txt' if mode == 'train' else ''
        self.pretrained_emb = None
        self.vocab_file = '{}/sst/vocab.txt'.format(self.dir) if vocab_file is None else vocab_file
        download(_get_dgl_url(_urls['sst']), path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/sst'.format(self.dir))
        self.trees = []
        self.num_classes = 5
        print('Preprocessing...')
        self._load()
        print('Dataset creation finished. #Trees:', len(self.trees))

    def _load(self):
        from nltk.corpus.reader import BracketParseCorpusReader
        # load vocab file
        self.vocab = OrderedDict()
        with open(self.vocab_file, encoding='utf-8') as vf:
            for line in vf.readlines():
                line = line.strip()
                self.vocab[line] = len(self.vocab)

        # filter glove
        if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
            glove_emb = {}
            with open(self.pretrained_file, 'r', encoding='utf-8') as pf:
                for line in pf.readlines():
                    sp = line.split(' ')
                    if sp[0].lower() in self.vocab:
                        glove_emb[sp[0].lower()] = np.array([float(x) for x in sp[1:]])
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader('{}/sst'.format(self.dir), files)
        sents = corpus.parsed_sents(files[0])

        #initialize with glove
        pretrained_emb = []
        fail_cnt = 0
        for line in self.vocab.keys():
            if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
                if not line.lower() in glove_emb:
                    fail_cnt += 1
                pretrained_emb.append(glove_emb.get(line.lower(), np.random.uniform(-0.05, 0.05, 300)))

        if self.pretrained_file != '' and os.path.exists(self.pretrained_file):
            self.pretrained_emb = F.tensor(np.stack(pretrained_emb, 0))
            print('Miss word in GloVe {0:.4f}'.format(1.0*fail_cnt/len(self.pretrained_emb)))
        # build trees
        for sent in sents:
            self.trees.append(self._build_tree(sent))


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
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['x', 'y', 'mask'])
        return ret

    def __getitem__(self, idx):
        """Get the tree with index idx.

        Parameters
        ----------
        idx : int
            Tree index.

        Returns
        -------
        dgl.DGLGraph
            Tree.
        """
        return self.trees[idx]

    def __len__(self):
        """Get the number of trees in the dataset.

        Returns
        -------
        int
            Number of trees.
        """
        return len(self.trees)

    @property
    def num_vocabs(self):
        return len(self.vocab)

    @staticmethod
    def batcher(device):
        def batcher_dev(batch):
            batch_trees = dgl.batch(batch)
            return SSTBatch(graph=batch_trees,
                            mask=batch_trees.ndata['mask'].to(device),
                            wordid=batch_trees.ndata['x'].to(device),
                            label=batch_trees.ndata['y'].to(device))
        return batcher_dev
