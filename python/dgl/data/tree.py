"""Tree-structured data.

Including:
    - SST
"""
from __future__ import absolute_import

from collections import namedtuple
from nltk.tree import Tree
from nltk.corpus.reader import BracketParseCorpusReader
import networkx as nx

import dgl
import dgl.backend as F
from dgl.data.utils import download, extract_archive, get_download_dir

_urls = {
    'sst' : 'https://www.dropbox.com/s/dw8kr2vuq7k4dqi/sst.zip?dl=1',
}

SSTBatch = namedtuple('SSTBatch', ['graph', 'nid_with_word', 'wordid', 'label'])

class SST(object):
    """SST"""
    PAD_WORD=-1
    def __init__(self, mode='train', vocab_file=None):
        self.mode = mode
        self.dir = get_download_dir()
        self.zip_file_path='{}/sst.zip'.format(self.dir)
        self.vocab_file = '{}/sst/vocab.txt'.format(self.dir) if vocab_file is None else vocab_file
        download(_urls['sst'], path=self.zip_file_path)
        extract_archive(self.zip_file_path, '{}/sst'.format(self.dir))
        self.trees = []
        self.num_classes = 5
        print('Preprocessing...')
        self._load()
        print('Dataset creation finished. #Trees:', len(self.trees))

    def _load(self):
        files = ['{}.txt'.format(self.mode)]
        corpus = BracketParseCorpusReader('{}/sst'.format(self.dir), files)
        sents = corpus.parsed_sents(files[0])
        # load vocab file
        self.vocab = {}
        with open(self.vocab_file) as vf:
            for line in vf.readlines():
                line = line.strip()
                self.vocab[line] = len(self.vocab)
        # build trees
        for sent in sents:
            self.trees.append(self._build_tree(sent))

    def _build_tree(self, root):
        g = nx.DiGraph()
        def _rec_build(nid, node):
            for child in node:
                cid = g.number_of_nodes()
                if isinstance(child[0], str):
                    # leaf node
                    word = self.vocab[child[0].lower()]
                    g.add_node(cid, x=word, y=int(child.label()))
                else:
                    g.add_node(cid, x=SST.PAD_WORD, y=int(child.label()))
                    _rec_build(cid, child)
                g.add_edge(cid, nid)
        # add root
        g.add_node(0, x=SST.PAD_WORD, y=int(root.label()))
        _rec_build(0, root)
        return dgl.DGLGraph(g)

    def __getitem__(self, idx):
        return self.trees[idx]

    def __len__(self):
        return len(self.trees)

    @property 
    def num_vocabs(self):
        return len(self.vocab)

    @staticmethod
    def batcher(batch):
        nid_with_word = []
        wordid = []
        label = []
        gnid = 0
        for tree in batch:
            for nid in range(tree.number_of_nodes()):
                if tree.nodes[nid]['x'] != SST.PAD_WORD:
                    nid_with_word.append(gnid)
                    wordid.append(tree.nodes[nid]['x'])
                label.append(tree.nodes[nid]['y'])
                gnid += 1
        batch_trees = dgl.batch(batch)
        return SSTBatch(graph=batch_trees,
                        nid_with_word=F.tensor(nid_with_word, dtype=F.int64),
                        wordid=F.tensor(wordid, dtype=F.int64),
                        label=F.tensor(label, dtype=F.int64))
