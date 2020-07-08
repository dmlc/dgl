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
from ..graph import DGLGraph
from ..graph import batch as graph_batch

class RGCNLinkDataset(DGLBuiltinDatasetm):
    """RGCN link prediction dataset

    The dataset contains a graph depicting the connectivity of a knowledge
    base. Currently, the knowledge bases from the
    `RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are
    FB15k-237, FB15k, wn18

    Parameters
    -----------
    name: str
        Name can be 'FB15k-237', 'FB15k' or 'wn18'.
    reverse: boo
        Whether add reverse edges. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    An object of this class has 5 member attributes needed for link
    prediction:

    num_nodes: int
        number of entities of knowledge base
    num_rels: int
        number of relations (including reverse relation) of knowledge base
    train: numpy.array
        all relation triplets (src, rel, dst) for training
    valid: numpy.array
        all relation triplets (src, rel, dst) for validation
    test: numpy.array
        all relation triplets (src, rel, dst) for testing

    Usually, user don't need to directly use this class. Instead, DGL provides
    wrapper function to load data (see example below).

    Examples
    --------
    Load FB15k-237 dataset

    >>> from dgl.contrib.data import load_data
    >>> data = load_data(dataset='FB15k-237')

    """
    def __init__(self, name, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        self.name = name
        self.reverse = reverse
        url = _get_dgl_url('dataset/') + '{}.tgz'.format(self.name)
        super(RGCNLinkDataset, self).__init__(name,
                                              url=url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

    def process(self, root_path):
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
            print("# training edges: {}".format(len(train)))
            print("# validation edges: {}".format(len(valid)))
            print("# testing edges: {}".format(len(test)))

        # for compatability
        self._train = train
        self._valid = valid
        self._test = test

        self._num_nodes = num_nodes
        self._num_rels = num_rels
        # build graph
        g = build_knowledge_graph(num_nodes, num_rels, train, valid, test, reverse=self.reverse)
        sekf._g = g

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
        save_graphs(str(graph_path), self.g)
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
        train_mask = self._g.ndata['train_mask'].numpy()
        val_mask = self._g.ndata['val_mask'].numpy()
        test_mask = self._g.ndata['test_mask'].numpy()
        self._g.ndata['train_edge_mask'] = generate_mask_tensor(self._g.ndata['train_edge_mask'].numpy())
        self._g.ndata['valid_edge_mask'] = generate_mask_tensor(self._g.ndata['valid_edge_mask'].numpy())
        self._g.ndata['test_edge_mask'] = generate_mask_tensor(self._g.ndata['test_edge_mask'].numpy())
        self._g.ndata['train_mask'] = generate_mask_tensor(train_mask)
        self._g.ndata['val_mask'] = generate_mask_tensor(val_mask)
        self._g.ndata['test_mask'] = generate_mask_tensor(test_mask)

        # for compatability
        etype = self.g.edata['etype'].numpy()
        u, v = self.g.all_edges(form='uv')
        u = u.numpy()
        v = v.numpy()
        train_idx = np.nonzero(train_mask==1)
        self._train = np.column_stack((u[train_idx], etype[train_idx], v[train_idx]))
        valid_idx = np.nonzero(valid_mask==1)
        self._valid = np.column_stack((u[valid_idx], etype[valid_idx], v[valid_idx]))
        test_idx = np.nonzero(test_mask==1)
        self._test = np.column_stack((u[test_idx], etype[test_idx], v[test_idx]))

        if self.verbose:
            print("# entities: {}".format(num_nodes))
            print("# relations: {}".format(num_rels))
            print("# training edges: {}".format(len(train_idx)))
            print("# validation edges: {}".format(len(valid_idx)))
            print("# testing edges: {}".format(len(test_idx)))

    @property
    def g(self):
        return self._g

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
            raw_reverse_sugb[re_type][0].append(d)
            raw_reverse_sugb[re_type][1].append(s)
            raw_reverse_subg_eset[e_type].append(edge_set)
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
        settype = np.concatenate([settype, np.full((settype.shape[0], -1), 0)])
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
    g = dgl.graph((s, d), num_nodes=num_nodes)
    etype = np.concatenate(fg_etype)
    settype = np.concatenate(fg_settype)
    g.edata['etype'] = F.tensor(etype, dtype=F.data_type_dict['long'])
    g.edata['train_edge_mask'] = train_edge_mask
    g.edata['valid_edge_mask'] = valid_edge_mask
    g.edata['test_edge_mask'] = test_edge_mask
    g.edata['train_mask'] = generate_mask_tensor(settype == 1) if reverse is True else train_edge_mask
    g.edata['valid_mask'] = generate_mask_tensor(settype == 2) if reverse is True else valid_edge_mask
    g.edata['test_mask'] = generate_mask_tensor(settype == 3) if reverse is True else test_edge_mask
    g.ndata['ntype'] = F.full_1d(num_nodes, 0, dtype=F.data_type_dict['long'], ctx=F.cpu())

    return g

class FB15k237Dataset(RGCNLinkDataset):
    r"""FB15k237 link prediction dataset

    FB15k-237 is a subset of FB15k where inverse 
    relations are removed. When creating the dataset,
    a reverse edge with reversed relation types are 
    created for each edge by default.
    
    Statistics
    ----------
    Nodes: xxx
    Edges: xxx
    Number of relation types: 237
    Number of reversed relation types: 237
    Label Split: Train: xxx ,Valid: xxx, Test: xxx

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
        
    Returns
    ----------
    FB15k237Dataset object with three properties:
        graph: A Homogeneous graph contain
            - edata['e_type']: edge relation type
            - edata['train_edge_mask']: positive training edge mask
            - edata['val_edge_mask']: positive validation edge mask
            - edata['test_edge_mask']: positive testing edge mask
            - edata['train_mask']: training edge set mask (include reversed training edges)
            - edata['val_mask']: validation edge set mask (include reversed validation edges)
            - edata['test_mask']: testing edge set mask (include reversed testing edges)
            - ndata['ntype']: node type. All 0 in this dataset
        num_nodes: Number of nodes
        num_rels: Number of relation types

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
    >>>
    """
    def __init__(self, reverse=True, raw_dir=None, force_reload=False, verbose=True):
        name = 'FB15k-237'
        super(FB15k237Dataset, self).__init__(name, raw_dir, force_reload, verbose)

class FB15kDataset(RGCNLinkDataset):
    r"""FB15k link prediction dataset

    The FB15K dataset was introduced in http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf, 
    It is a subset of Freebase which contains about 
    14,951 entities with 1,345 different relations.
    When creating the dataset, a reverse edge with 
    reversed relation types are created for each edge 
    by default.
    
    Statistics
    ----------
    Nodes: 14,951
    Edges: xxx
    Number of relation types: 1,345
    Number of reversed relation types: 1,345
    Label Split: Train: xxx ,Valid: xxx, Test: xxx

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
        
    Returns
    --------
    FB15kDataset object with three properties:
        graph: A Homogeneous graph contain
            - edata['e_type']: edge relation type
            - edata['train_edge_mask']: positive training edge mask
            - edata['val_edge_mask']: positive validation edge mask
            - edata['test_edge_mask']: positive testing edge mask
            - edata['train_mask']: training edge set mask (include reversed training edges)
            - edata['val_mask']: validation edge set mask (include reversed validation edges)
            - edata['test_mask']: testing edge set mask (include reversed testing edges)
            - ndata['ntype']: node type. All 0 in this dataset
        num_nodes: Number of nodes
        num_rels: Number of relation types

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
        super(FB15kDataset, self).__init__(name, raw_dir, force_reload, verbose)

class WN18Dataset(RGCNLinkDataset):
    r""" WN18 dataset.
    
    The WN18 dataset was introduced in http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf, 
    It included the full 18 relations scraped from
    WordNet for roughly 41,000 synsets. When creating 
    the dataset, a reverse edge with reversed relation 
    types are created for each edge by default.
    
    Statistics
    ----------
    Nodes: xxx
    Edges: xxx
    Number of relation types: 18
    Number of reversed relation types: 18
    Label Split: Train: xxx ,Valid: xxx, Test: xxx
    
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

    Returns
    ----------
    WN18Dataset object with three properties:
        graph: A Homogeneous graph contain
            - edata['e_type']: edge relation type
            - edata['train_edge_mask']: positive training edge mask
            - edata['val_edge_mask']: positive validation edge mask
            - edata['test_edge_mask']: positive testing edge mask
            - edata['train_mask']: training edge set mask (include reversed training edges)
            - edata['val_mask']: validation edge set mask (include reversed validation edges)
            - edata['test_mask']: testing edge set mask (include reversed testing edges)
            - ndata['ntype']: node type. All 0 in this dataset
        num_nodes: Number of nodes
        num_rels: Number of relation types

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
        super(WN18Dataset, self).__init__(name, raw_dir, force_reload, verbose)

