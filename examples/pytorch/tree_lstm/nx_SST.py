import numpy as NP
from nltk.tree import Tree
from nltk.corpus.reader import BracketParseCorpusReader as CorpusReader
import networkx as nx

class nx_BCT_Reader(object):
    # Binary Constituency Tree constructor for networkx
    def __init__(self, fnames=['trees/train.txt', 'trees/dev.txt', 'trees/test.txt']):
        # fnames must be three items which means the file path of train, validation, test set, respectively.
        self.corpus = CorpusReader('.', fnames)
        self.train = self.corpus.parsed_sents(fnames[0])
        self.dev = self.corpus.parsed_sents(fnames[1])
        self.test = self.corpus.parsed_sents(fnames[2])

    def create_BCT(self, root):
        self.node_cnt = 0
        self.G = nx.DiGraph()
        def _rec(node, nx_node, depth=0):
            for child in node:
                node_id = str(self.node_cnt) + '_' + str(depth+1)
                self.node_cnt += 1
                if isinstance(child[0], str) or isinstance(child[0], unicode):
                    self.G.add_node(node_id, word=child[0], label=None)
                else:
                    self.G.add_node(node_id, word=None, lable=child.label())
                    if isinstance(child, Tree): #check illegal trees
                        _rec(child, node_id)
                self.G.add_edge(node_id, nx_node)

        self.G.add_node('0_0', word=None) # add root into nx Graph
        _rec(root, '0_0')

        return self.G

    def generator(self, mode='train'):
        assert mode in ['train', 'dev', 'test']
        for s in self.__dict__[mode]:
            yield self.create_BCT(s)

