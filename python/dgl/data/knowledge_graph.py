from __future__ import absolute_import

import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, sys

from .dgl_dataset import DGLBuiltinDataset
from .utils import download, extract_archive, get_download_dir
from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor
from .utils import deprecate_property, deprecate_function
from ..utils import retry_method_with_fix
from .. import backend as F
from ..convert import graph as dgl_graph

class KnowledgeGraphDataset(DGLBuiltinDataset):
    """KnowledgeGraph link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    Parameters
    -----------
    name: str
        Name can be 'FB15k-237', 'FB15k' or 'wn18'.
    reverse: bool
        Whether add reverse edges. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    """
    def __init__(self, name, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        self._name = name
        self.reverse = reverse
        url = _get_dgl_url('dataset/') + '{}.tgz'.format(name)
        super(KnowledgeGraphDataset, self).__init__(name,
                                                    url=url,
                                                    raw_dir=raw_dir,
                                                    force_reload=force_reload,
                                                    verbose=verbose)

    def download(self):
        r""" Automatically download data and extract it.
        """
        tgz_path = os.path.join(self.raw_dir, self.name + '.tgz')
        download(self.url, path=tgz_path)
        extract_archive(tgz_path, self.raw_path)

    def process(self):
        """
        The original knowledge base is stored in triplets.
        This function will parse these triplets and build the DGLGraph.
        """
        root_path = self.raw_path
        entity_path = os.path.join(root_path, 'entities.dict')
        relation_path = os.path.join(root_path, 'relations.dict')
        train_path = os.path.join(root_path, 'train.txt')
        valid_path = os.path.join(root_path, 'valid.txt')
        test_path = os.path.join(root_path, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        train = np.asarray(_read_triplets_as_list(train_path, entity_dict, relation_dict))
        valid = np.asarray(_read_triplets_as_list(valid_path, entity_dict, relation_dict))
        test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))
        num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        if self.verbose:
            print("# entities: {}".format(num_nodes))
            print("# relations: {}".format(num_rels))
            print("# training edges: {}".format(train.shape[0]))
            print("# validation edges: {}".format(valid.shape[0]))
            print("# testing edges: {}".format(test.shape[0]))

        # for compatability
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        # build graph
        g, data = build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=self.reverse)
        etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, val_mask, test_mask = data
        g.edata['train_edge_mask'] = train_edge_mask
        g.edata['valid_edge_mask'] = valid_edge_mask
        g.edata['test_edge_mask'] = test_edge_mask
        g.edata['train_mask'] = train_mask
        g.edata['val_mask'] = val_mask
        g.edata['test_mask'] = test_mask
        g.edata['etype'] = etype
        g.ndata['ntype'] = ntype
        self._g = g

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        return self._g

    def __len__(self):
        return 1

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self._g)
        save_info(str(info_path), {'num_nodes': self.num_nodes,
                                   'num_rels': self.num_rels})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._num_nodes = info['num_nodes']
        self._num_rels = info['num_rels']
        self._g = graphs[0]
        train_mask = self._g.edata['train_edge_mask'].numpy()
        val_mask = self._g.edata['valid_edge_mask'].numpy()
        test_mask = self._g.edata['test_edge_mask'].numpy()

        # convert mask tensor into bool tensor if possible
        self._g.edata['train_edge_mask'] = generate_mask_tensor(self._g.edata['train_edge_mask'].numpy())
        self._g.edata['valid_edge_mask'] = generate_mask_tensor(self._g.edata['valid_edge_mask'].numpy())
        self._g.edata['test_edge_mask'] = generate_mask_tensor(self._g.edata['test_edge_mask'].numpy())
        self._g.edata['train_mask'] = generate_mask_tensor(self._g.edata['train_mask'].numpy())
        self._g.edata['val_mask'] = generate_mask_tensor(self._g.edata['val_mask'].numpy())
        self._g.edata['test_mask'] = generate_mask_tensor(self._g.edata['test_mask'].numpy())

        # for compatability (with 0.4.x) generate train_idx, valid_idx and test_idx
        etype = self._g.edata['etype'].numpy()
        self._etype = etype
        u, v = self._g.all_edges(form='uv')
        u = u.numpy()
        v = v.numpy()
        train_idx = np.nonzero(train_mask==1)
        self._train = np.column_stack((u[train_idx], etype[train_idx], v[train_idx]))
        valid_idx = np.nonzero(val_mask==1)
        self._valid = np.column_stack((u[valid_idx], etype[valid_idx], v[valid_idx]))
        test_idx = np.nonzero(test_mask==1)
        self._test = np.column_stack((u[test_idx], etype[test_idx], v[test_idx]))

        if self.verbose:
            print("# entities: {}".format(self.num_nodes))
            print("# relations: {}".format(self.num_rels))
            print("# training edges: {}".format(self._train.shape[0]))
            print("# validation edges: {}".format(self._valid.shape[0]))
            print("# testing edges: {}".format(self._test.shape[0]))

    @property
    def num_nodes(self):
        return self._num_nodes

    @property
    def num_rels(self):
        return self._num_rels

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def train(self):
        deprecate_property('dataset.train', 'g.edata[\'train_mask\']')
        return self._train

    @property
    def valid(self):
        deprecate_property('dataset.valid', 'g.edata[\'val_mask\']')
        return self._valid

    @property
    def test(self):
        deprecate_property('dataset.test', 'g.edata[\'test_mask\']')
        return self._test

def _read_dictionary(filename):
    d = {}
    with open(filename, 'r+') as f:
        for line in f:
            line = line.strip().split('\t')
            d[line[1]] = int(line[0])
    return d

def _read_triplets(filename):
    with open(filename, 'r+') as f:
        for line in f:
            processed_line = line.strip().split('\t')
            yield processed_line

def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l

def build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=True):
    """ Create a DGL Homogeneous graph with heterograph info stored as node or edge features.
    """
    src = []
    rel = []
    dst = []
    raw_subg = {}
    raw_subg_eset = {}
    raw_subg_etype = {}
    raw_reverse_sugb = {}
    raw_reverse_subg_eset = {}
    raw_reverse_subg_etype = {}

    # here there is noly one node type
    s_type = "node"
    d_type = "node"

    def add_edge(s, r, d, reverse, edge_set):
        r_type = str(r)
        e_type = (s_type, r_type, d_type)
        if raw_subg.get(e_type, None) is None:
            raw_subg[e_type] = ([], [])
            raw_subg_eset[e_type] = []
            raw_subg_etype[e_type] = []
        raw_subg[e_type][0].append(s)
        raw_subg[e_type][1].append(d)
        raw_subg_eset[e_type].append(edge_set)
        raw_subg_etype[e_type].append(r)

        if reverse is True:
            r_type = str(r + num_rels)
            re_type = (d_type, r_type, s_type)
            if raw_reverse_sugb.get(re_type, None) is None:
                raw_reverse_sugb[re_type] = ([], [])
                raw_reverse_subg_etype[re_type] = []
                raw_reverse_subg_eset[re_type] = []
            raw_reverse_sugb[re_type][0].append(d)
            raw_reverse_sugb[re_type][1].append(s)
            raw_reverse_subg_eset[re_type].append(edge_set)
            raw_reverse_subg_etype[re_type].append(r + num_rels)

    for edge in train:
        s, r, d = edge
        assert r < num_rels
        add_edge(s, r, d, reverse, 1) # train set

    for edge in valid:
        s, r, d = edge
        assert r < num_rels
        add_edge(s, r, d, reverse, 2) # valid set

    for edge in test:
        s, r, d = edge
        assert r < num_rels
        add_edge(s, r, d, reverse, 3) # test set

    subg = []
    fg_s = []
    fg_d = []
    fg_etype = []
    fg_settype = []
    for e_type, val in raw_subg.items():
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)
        etype = raw_subg_etype[e_type]
        etype = np.asarray(etype)
        settype = raw_subg_eset[e_type]
        settype = np.asarray(settype)

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    settype = np.concatenate(fg_settype)
    if reverse is True:
        settype = np.concatenate([settype, np.full((settype.shape[0]), 0)])
    train_edge_mask = generate_mask_tensor(settype == 1)
    valid_edge_mask = generate_mask_tensor(settype == 2)
    test_edge_mask = generate_mask_tensor(settype == 3)

    for e_type, val in raw_reverse_sugb.items():
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)
        etype = raw_reverse_subg_etype[e_type]
        etype = np.asarray(etype)
        settype = raw_reverse_subg_eset[e_type]
        settype = np.asarray(settype)

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    s = np.concatenate(fg_s)
    d = np.concatenate(fg_d)
    g = dgl_graph((s, d), num_nodes=num_nodes)
    etype = np.concatenate(fg_etype)
    settype = np.concatenate(fg_settype)
    etype = F.tensor(etype, dtype=F.data_type_dict['int64'])
    train_edge_mask = train_edge_mask
    valid_edge_mask = valid_edge_mask
    test_edge_mask = test_edge_mask
    train_mask = generate_mask_tensor(settype == 1) if reverse is True else train_edge_mask
    valid_mask = generate_mask_tensor(settype == 2) if reverse is True else valid_edge_mask
    test_mask = generate_mask_tensor(settype == 3) if reverse is True else test_edge_mask
    ntype = F.full_1d(num_nodes, 0, dtype=F.data_type_dict['int64'], ctx=F.cpu())

    return g, (etype, ntype, train_edge_mask, valid_edge_mask, test_edge_mask, train_mask, valid_mask, test_mask)

class FB15k237Dataset(KnowledgeGraphDataset):
    r"""FB15k237 link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = FB15k237Dataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    FB15k-237 is a subset of FB15k where inverse
    relations are removed. When creating the dataset,
    a reverse edge with reversed relation types are
    created for each edge by default.

    FB15k237 dataset statistics:

    - Nodes: 14541
    - Number of relation types: 237
    - Number of reversed relation types: 237
    - Label Split:

        - Train: 272115
        - Valid: 17535
        - Test: 20466

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = FB15k237Dataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>> test_mask = g.edata['test_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  preserve_nodes=True)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                preserve_nodes=True)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        name = 'FB15k-237'
        super(FB15k237Dataset, self).__init__(name, reverse, raw_dir, force_reload, verbose)

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, FB15k237Dataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        return super(FB15k237Dataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(FB15k237Dataset, self).__len__()

class FB15kDataset(KnowledgeGraphDataset):
    r"""FB15k link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = FB15kDataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    The FB15K dataset was introduced in `Translating Embeddings for Modeling
    Multi-relational Data <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
    It is a subset of Freebase which contains about
    14,951 entities with 1,345 different relations.
    When creating the dataset, a reverse edge with
    reversed relation types are created for each edge
    by default.

    FB15k dataset statistics:

    - Nodes: 14,951
    - Number of relation types: 1,345
    - Number of reversed relation types: 1,345
    - Label Split:

        - Train: 483142
        - Valid: 50000
        - Test: 59071

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = FB15kDataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  preserve_nodes=True)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                preserve_nodes=True)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    >>>
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        name = 'FB15k'
        super(FB15kDataset, self).__init__(name, reverse, raw_dir, force_reload, verbose)

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, FB15kDataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        return super(FB15kDataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(FB15kDataset, self).__len__()

class WN18Dataset(KnowledgeGraphDataset):
    r""" WN18 link prediction dataset.

    .. deprecated:: 0.5.0

        - ``train`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> train_mask = graph.edata['train_mask']
            >>> train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(train_idx)
            >>> rel = graph.edata['etype'][train_idx]

        - ``valid`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> val_mask = graph.edata['val_mask']
            >>> val_idx = th.nonzero(val_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(val_idx)
            >>> rel = graph.edata['etype'][val_idx]

        - ``test`` is deprecated, it is replaced by:

            >>> dataset = WN18Dataset()
            >>> graph = dataset[0]
            >>> test_mask = graph.edata['test_mask']
            >>> test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()
            >>> src, dst = graph.edges(test_idx)
            >>> rel = graph.edata['etype'][test_idx]

    The WN18 dataset was introduced in `Translating Embeddings for Modeling
    Multi-relational Data <http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf>`_.
    It included the full 18 relations scraped from
    WordNet for roughly 41,000 synsets. When creating
    the dataset, a reverse edge with reversed relation
    types are created for each edge by default.

    WN18 dataset statistics:

    - Nodes: 40943
    - Number of relation types: 18
    - Number of reversed relation types: 18
    - Label Split:

        - Train: 141442
        - Valid: 5000
        - Test: 5000

    Parameters
    ----------
    reverse : bool
        Whether to add reverse edge. Default True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
        Whether to print out progress information. Default: True.

    Attributes
    ----------
    num_nodes: int
        Number of nodes
    num_rels: int
        Number of relation types
    train: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the training graph
    valid: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the validation graph
    test: numpy.ndarray
        A numpy array of triplets (src, rel, dst) for the test graph

    Examples
    ----------
    >>> dataset = WN18Dataset()
    >>> g = dataset.graph
    >>> e_type = g.edata['e_type']
    >>>
    >>> # get data split
    >>> train_mask = g.edata['train_mask']
    >>> val_mask = g.edata['val_mask']
    >>>
    >>> train_set = th.arange(g.number_of_edges())[train_mask]
    >>> val_set = th.arange(g.number_of_edges())[val_mask]
    >>>
    >>> # build train_g
    >>> train_edges = train_set
    >>> train_g = g.edge_subgraph(train_edges,
                                  preserve_nodes=True)
    >>> train_g.edata['e_type'] = e_type[train_edges];
    >>>
    >>> # build val_g
    >>> val_edges = th.cat([train_edges, val_edges])
    >>> val_g = g.edge_subgraph(val_edges,
                                preserve_nodes=True)
    >>> val_g.edata['e_type'] = e_type[val_edges];
    >>>
    >>> # Train, Validation and Test
    >>>
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        name = 'wn18'
        super(WN18Dataset, self).__init__(name, reverse, raw_dir, force_reload, verbose)

    def __getitem__(self, idx):
        r"""Gets the graph object

        Parameters
        -----------
        idx: int
            Item index, WN18Dataset has only one graph object

        Return
        -------
        :class:`dgl.DGLGraph`

            The graph contains

            - ``edata['e_type']``: edge relation type
            - ``edata['train_edge_mask']``: positive training edge mask
            - ``edata['val_edge_mask']``: positive validation edge mask
            - ``edata['test_edge_mask']``: positive testing edge mask
            - ``edata['train_mask']``: training edge set mask (include reversed training edges)
            - ``edata['val_mask']``: validation edge set mask (include reversed validation edges)
            - ``edata['test_mask']``: testing edge set mask (include reversed testing edges)
            - ``ndata['ntype']``: node type. All 0 in this dataset
        """
        return super(WN18Dataset, self).__getitem__(idx)

    def __len__(self):
        r"""The number of graphs in the dataset."""
        return super(WN18Dataset, self).__len__()

def load_data(dataset):
    r"""Load knowledge graph dataset for RGCN link prediction tasks

    It supports three datasets: wn18, FB15k and FB15k-237

    Parameters
    ----------
    dataset: str
        The name of the dataset to load.

    Return
    ------
    The dataset object.
    """
    if dataset == 'wn18':
        return WN18Dataset()
    elif dataset == 'FB15k':
        return FB15kDataset()
    elif dataset == 'FB15k-237':
        return FB15k237Dataset()
