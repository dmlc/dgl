"""RDF datasets
Datasets from "A Collection of Benchmark Datasets for
Systematic Evaluations of Machine Learning on
the Semantic Web"
"""
import os
from collections import OrderedDict
import itertools
import rdflib as rdf
import abc
import re

import networkx as nx
import numpy as np

import dgl
import dgl.backend as F
from .dgl_dataset import DGLBuiltinDataset
from .utils import download, extract_archive, get_download_dir
from .utils import save_graphs, load_graphs, save_info, load_info, makedirs, _get_dgl_url
from .utils import generate_mask_tensor, idx2mask, deprecate_property, deprecate_class
from ..utils import retry_method_with_fix

__all__ = ['AIFB', 'MUTAG', 'BGS', 'AM', 'AIFBDataset', 'MUTAGDataset', 'BGSDataset', 'AMDataset']

# Dictionary for renaming reserved node/edge type names to the ones
# that are allowed by nn.Module.
RENAME_DICT = {
    'type' : 'rdftype',
    'rev-type' : 'rev-rdftype',
}

class Entity:
    """Class for entities
    Parameters
    ----------
    id : str
        ID of this entity
    cls : str
        Type of this entity
    """
    def __init__(self, id, cls):
        self.id = id
        self.cls = cls

    def __str__(self):
        return '{}/{}'.format(self.cls, self.id)

class Relation:
    """Class for relations
    Parameters
    ----------
    cls : str
        Type of this relation
    """
    def __init__(self, cls):
        self.cls = cls

    def __str__(self):
        return str(self.cls)

class RDFGraphDataset(DGLBuiltinDataset):
    """Base graph dataset class from RDF tuples.

    To derive from this, implement the following abstract methods:
    * ``parse_entity``
    * ``parse_relation``
    * ``process_tuple``
    * ``process_idx_file_line``
    * ``predict_category``
    Preprocessed graph and other data will be cached in the download folder
    to speedup data loading.
    The dataset should contain a "trainingSet.tsv" and a "testSet.tsv" file
    for training and testing samples.

    Attributes
    ----------
    graph : dgl.DGLHeteroGraph
        Graph structure
    num_classes : int
        Number of classes to predict
    predict_category : str
        The entity category (node type) that has labels for prediction
    train_idx : Tensor
        Entity IDs for training. All IDs are local IDs w.r.t. to ``predict_category``.
    test_idx : Tensor
        Entity IDs for testing. All IDs are local IDs w.r.t. to ``predict_category``.
    labels : Tensor
        All the labels of the entities in ``predict_category``

    Parameters
    ----------
    name : str
        Name of the dataset
    url : str or path
        URL to download the raw dataset.
    predict_category : str
        Predict category.
    print_every : int, optional
        Log for every X tuples.
    insert_reverse : bool, optional
        If true, add reverse edge and reverse relations to the final graph.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool, optional
        If true, force load and process from raw data. Ignore cached pre-processed data.
    verbose: bool
        Whether to print out progress information. Default: True.
    """
    def __init__(self, name, url, predict_category,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        self._insert_reverse = insert_reverse
        self._print_every = print_every
        self._predict_category = predict_category

        super(RDFGraphDataset, self).__init__(name, url,
                                              raw_dir=raw_dir,
                                              force_reload=force_reload,
                                              verbose=verbose)

    def process(self):
        raw_tuples = self.load_raw_tuples(self.raw_path)
        self.process_raw_tuples(raw_tuples, self.raw_path)

    def load_raw_tuples(self, root_path):
        raw_rdf_graphs = []
        for i, filename in enumerate(os.listdir(root_path)):
            fmt = None
            if filename.endswith('nt'):
                fmt = 'nt'
            elif filename.endswith('n3'):
                fmt = 'n3'
            if fmt is None:
                continue
            g = rdf.Graph()
            print('Parsing file %s ...' % filename)
            g.parse(os.path.join(root_path, filename), format=fmt)
            raw_rdf_graphs.append(g)
        return itertools.chain(*raw_rdf_graphs)

    def process_raw_tuples(self, raw_tuples, root_path):
        mg = nx.MultiDiGraph()
        ent_classes = OrderedDict()
        rel_classes = OrderedDict()
        entities = OrderedDict()
        src = []
        dst = []
        ntid = []
        etid = []
        sorted_tuples = []
        for t in raw_tuples:
            sorted_tuples.append(t)
        sorted_tuples.sort()

        for i, (sbj, pred, obj) in enumerate(sorted_tuples):
            if self.verbose and i % self._print_every == 0:
                print('Processed %d tuples, found %d valid tuples.' % (i, len(src)))
            sbjent = self.parse_entity(sbj)
            rel = self.parse_relation(pred)
            objent = self.parse_entity(obj)
            processed = self.process_tuple((sbj, pred, obj), sbjent, rel, objent)
            if processed is None:
                # ignored
                continue
            # meta graph
            sbjclsid = _get_id(ent_classes, sbjent.cls)
            objclsid = _get_id(ent_classes, objent.cls)
            relclsid = _get_id(rel_classes, rel.cls)
            mg.add_edge(sbjent.cls, objent.cls, key=rel.cls)
            if self._insert_reverse:
                mg.add_edge(objent.cls, sbjent.cls, key='rev-%s' % rel.cls)
            # instance graph
            src_id = _get_id(entities, str(sbjent))
            if len(entities) > len(ntid):  # found new entity
                ntid.append(sbjclsid)
            dst_id = _get_id(entities, str(objent))
            if len(entities) > len(ntid):  # found new entity
                ntid.append(objclsid)
            src.append(src_id)
            dst.append(dst_id)
            etid.append(relclsid)

        src = np.asarray(src)
        dst = np.asarray(dst)
        ntid = np.asarray(ntid)
        etid = np.asarray(etid)
        ntypes = list(ent_classes.keys())
        etypes = list(rel_classes.keys())

        # add reverse edge with reverse relation
        if self._insert_reverse:
            if self.verbose:
                print('Adding reverse edges ...')
            newsrc = np.hstack([src, dst])
            newdst = np.hstack([dst, src])
            src = newsrc
            dst = newdst
            etid = np.hstack([etid, etid + len(etypes)])
            etypes.extend(['rev-%s' % t for t in etypes])

        hg = self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)

        if self.verbose:
            print('Load training/validation/testing split ...')
        idmap = F.asnumpy(hg.nodes[self.predict_category].data[dgl.NID])
        glb2lcl = {glbid : lclid for lclid, glbid in enumerate(idmap)}
        def findidfn(ent):
            if ent not in entities:
                return None
            else:
                return glb2lcl[entities[ent]]
        self._hg = hg
        train_idx, test_idx, labels, num_classes = self.load_data_split(findidfn, root_path)

        train_mask = idx2mask(train_idx, self._hg.number_of_nodes(self.predict_category))
        test_mask = idx2mask(test_idx, self._hg.number_of_nodes(self.predict_category))
        labels = F.tensor(labels).long()

        self._hg.nodes[self.predict_category].data['train_mask'] = generate_mask_tensor(train_mask)
        self._hg.nodes[self.predict_category].data['test_mask'] = generate_mask_tensor(test_mask)
        self._hg.nodes[self.predict_category].data['labels'] = labels
        # save for compatability
        self._train_idx = F.tensor(train_idx)
        self._test_idx = F.tensor(test_idx)
        self._labels = labels
        self._num_classes = num_classes

    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        # create homo graph
        if self.verbose:
            print('Creating one whole graph ...')
        g = dgl.graph((src, dst))
        g.ndata[dgl.NTYPE] = F.tensor(ntid)
        g.edata[dgl.ETYPE] = F.tensor(etid)
        if self.verbose:
            print('Total #nodes:', g.number_of_nodes())
            print('Total #edges:', g.number_of_edges())

        # rename names such as 'type' so that they an be used as keys
        # to nn.ModuleDict
        etypes = [RENAME_DICT.get(ty, ty) for ty in etypes]
        mg_edges = mg.edges(keys=True)
        mg = nx.MultiDiGraph()
        for sty, dty, ety in mg_edges:
            mg.add_edge(sty, dty, key=RENAME_DICT.get(ety, ety))

        # convert to heterograph
        if self.verbose:
            print('Convert to heterograph ...')
        hg = dgl.to_hetero(g,
                           ntypes,
                           etypes,
                           metagraph=mg)
        if self.verbose:
            print('#Node types:', len(hg.ntypes))
            print('#Canonical edge types:', len(hg.etypes))
            print('#Unique edge type names:', len(set(hg.etypes)))
        return hg

    def load_data_split(self, ent2id, root_path):
        label_dict = {}
        labels = np.zeros((self.graph.number_of_nodes(self.predict_category),)) - 1
        train_idx = self.parse_idx_file(
            os.path.join(root_path, 'trainingSet.tsv'),
            ent2id, label_dict, labels)
        test_idx = self.parse_idx_file(
            os.path.join(root_path, 'testSet.tsv'),
            ent2id, label_dict, labels)
        train_idx = np.array(train_idx)
        test_idx = np.array(test_idx)
        labels = np.array(labels)
        num_classes = len(label_dict)
        return train_idx, test_idx, labels, num_classes

    def parse_idx_file(self, filename, ent2id, label_dict, labels):
        idx = []
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue  # first line is the header
                sample, label = self.process_idx_file_line(line)
                #person, _, label = line.strip().split('\t')
                ent = self.parse_entity(sample)
                entid = ent2id(str(ent))
                if entid is None:
                    print('Warning: entity "%s" does not have any valid links associated. Ignored.' % str(ent))
                else:
                    idx.append(entid)
                    lblid = _get_id(label_dict, label)
                    labels[entid] = lblid
        return idx

    def has_cache(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        if os.path.exists(graph_path) and \
            os.path.exists(info_path):
            return True

        return False

    def save(self):
        """save the graph list and the labels"""
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        save_graphs(str(graph_path), self.graph)
        save_info(str(info_path), {'num_classes': self.num_classes,
                                   'predict_category': self.predict_category})

    def load(self):
        graph_path = os.path.join(self.save_path,
                                  self.save_name + '.bin')
        info_path = os.path.join(self.save_path,
                                 self.save_name + '.pkl')
        graphs, _ = load_graphs(str(graph_path))

        info = load_info(str(info_path))
        self._num_classes = info['num_classes']
        self._predict_category = info['predict_category']
        self._hg = graphs[0]
        train_mask = self._hg.nodes[self.predict_category].data['train_mask']
        test_mask = self._hg.nodes[self.predict_category].data['test_mask']

        train_idx = F.nonzero_1d(train_mask)
        test_idx = F.nonzero_1d(test_mask)
        self._train_idx = train_idx
        self._test_idx = test_idx
        self._labels = self._hg.nodes[self.predict_category].data['labels']

    @property
    def save_name(self):
        return self.name + '_dgl_graph'

    @property
    def graph(self):
        return self._hg

    @property
    def predict_category(self):
        return self._predict_category

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def train_idx(self):
        deprecate_property('dataset.train_idx', 'train_mask = g.ndata[\'train_mask\']')
        return self._train_idx

    @property
    def test_idx(self):
        deprecate_property('dataset.test_idx', 'train_mask = g.ndata[\'test_mask\']')
        return self._test_idx

    @property
    def labels(self):
        deprecate_property('dataset.labels', 'train_mask = g.ndata[\'labels\']')
        return self._labels

    @abc.abstractmethod
    def parse_entity(self, term):
        """Parse one entity from an RDF term.
        Return None if the term does not represent a valid entity and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Entity or None
            An entity.
        """
        pass

    @abc.abstractmethod
    def parse_relation(self, term):
        """Parse one relation from an RDF term.
        Return None if the term does not represent a valid relation and the
        whole tuple should be ignored.
        Parameters
        ----------
        term : rdflib.term.Identifier
            RDF term
        Returns
        -------
        Relation or None
            A relation
        """
        pass

    @abc.abstractmethod
    def process_tuple(self, raw_tuple, sbj, rel, obj):
        """Process the tuple.
        Return (Entity, Relation, Entity) tuple for as the final tuple.
        Return None if the tuple should be ignored.

        Parameters
        ----------
        raw_tuple : tuple of rdflib.term.Identifier
            (subject, predicate, object) tuple
        sbj : Entity
            Subject entity
        rel : Relation
            Relation
        obj : Entity
            Object entity
        Returns
        -------
        (Entity, Relation, Entity)
            The final tuple or None if should be ignored
        """
        pass

    @abc.abstractmethod
    def process_idx_file_line(self, line):
        """Process one line of ``trainingSet.tsv`` or ``testSet.tsv``.
        Parameters
        ----------
        line : str
            One line of the file
        Returns
        -------
        (str, str)
            One sample and its label
        """
        pass

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def save_strlist(filename, strlist):
    with open(filename, 'w') as f:
        for s in strlist:
            f.write(s + '\n')

def load_strlist(filename):
    with open(filename, 'r') as f:
        ret = []
        for line in f:
            ret.append(line.strip())
        return ret

class AIFBDataset(RDFGraphDataset):
    """AIFB dataset.
     AIFB DataSet is a Semantic Web (RDF) dataset used as a benchmark in
     data mining.  It records the organizational structure of AIFB at the
     University of Karlsruhe.
    Statistics
    ===
    Nodes: 19717
    Edges: 88651
    Node Types: xxx
    Edge Types: xxx
    Target Category:
    Number of Classes: xx
    Label Split: Train: xx ,Valid: xx, Test: xx
    Parameters
    -----------
    print_every: int
        Log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    Returns
    ===
    AIFBDataset object with three properties:
        graph: A Heterogenous graph containing the
            graph structure, node features and labels.
        predict_category: The category name to run the node classification
            prediction.
        num_of_class: number of publication categories
            for the classification task.

    Examples
    --------
    >>> dataset = dgl.data.rdf.AIFBDataset()
    >>> print(dataset.graph)
    """

    employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
    affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")
    entity_prefix = 'http://www.aifb.uni-karlsruhe.de/'
    relation_prefix = 'http://swrc.ontoware.org/'

    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        url = _get_dgl_url('dataset/rdf/aifb-hetero.zip')
        name = 'aifb-hetero'
        predict_category = 'Personen'
        super(AIFBDataset, self).__init__(name, url, predict_category,
                                          print_every=print_every,
                                          insert_reverse=insert_reverse,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose)

    def parse_entity(self, term):
        if isinstance(term, rdf.Literal):
            return Entity(id=str(term), cls="_Literal")
        if isinstance(term, rdf.BNode):
            return None
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            sp = entstr.split('/')
            return Entity(id=sp[5], cls=sp[3])
        else:
            return None

    def parse_relation(self, term):
        if term == self.employs or term == self.affiliation:
            return None
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            return Relation(cls=relstr.split('/')[3])
        else:
            relstr = relstr.split('/')[-1]
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        person, _, label = line.strip().split('\t')
        return person, label

class AIFB(AIFBDataset):
    """AIFB dataset. Same as AIFBDataset.
    """
    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        deprecate_class('AIFB', 'AIFBDataset')
        super(AIFB, self).__init__(print_every,
                                   insert_reverse,
                                   raw_dir,
                                   force_reload,
                                   verbose)


class MUTAGDataset(RDFGraphDataset):
    """MUTAG dataset.
    Statistics
    ===
    Nodes: 19717
    Edges: 88651
    Node Types: xxx
    Edge Types: xxx
    Target Category:
    Number of Classes:
    Label Split: Train: xx ,Valid: xx, Test: xx
    Parameters
    -----------
    print_every: int
        Log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    Returns
    ===
    MUTAGDataset object with three properties:
        graph: A Heterogenous graph containing the
            graph structure, node features and labels.
        predict_category: The category name to run the node classification
            prediction.
        num_of_class: number of publication categories
            for the classification task.

    Examples
    --------
    >>> dataset = dgl.data.rdf.MUTAG()
    >>> print(dataset.graph)
    """

    d_entity = re.compile("d[0-9]")
    bond_entity = re.compile("bond[0-9]")

    is_mutagenic = rdf.term.URIRef("http://dl-learner.org/carcinogenesis#isMutagenic")
    rdf_type = rdf.term.URIRef("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    rdf_subclassof = rdf.term.URIRef("http://www.w3.org/2000/01/rdf-schema#subClassOf")
    rdf_domain = rdf.term.URIRef("http://www.w3.org/2000/01/rdf-schema#domain")

    entity_prefix = 'http://dl-learner.org/carcinogenesis#'
    relation_prefix = entity_prefix

    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        url = _get_dgl_url('dataset/rdf/mutag-hetero.zip')
        name = 'mutag-hetero'
        predict_category = 'd'
        super(MUTAGDataset, self).__init__(name, url, predict_category,
                                           print_every=print_every,
                                           insert_reverse=insert_reverse,
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)

    def parse_entity(self, term):
        if isinstance(term, rdf.Literal):
            return Entity(id=str(term), cls="_Literal")
        elif isinstance(term, rdf.BNode):
            return None
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            inst = entstr[len(self.entity_prefix):]
            if self.d_entity.match(inst):
                cls = 'd'
            elif self.bond_entity.match(inst):
                cls = 'bond'
            else:
                cls = None
            return Entity(id=inst, cls=cls)
        else:
            return None

    def parse_relation(self, term):
        if term == self.is_mutagenic:
            return None
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            cls = relstr[len(self.relation_prefix):]
            return Relation(cls=cls)
        else:
            relstr = relstr.split('/')[-1]
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None

        if not raw_tuple[1].startswith('http://dl-learner.org/carcinogenesis#'):
            obj.cls = 'SCHEMA'
            if sbj.cls is None:
                sbj.cls = 'SCHEMA'
        if obj.cls is None:
            obj.cls = rel.cls

        assert sbj.cls is not None and obj.cls is not None

        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        bond, _, label = line.strip().split('\t')
        return bond, label

class MUTAG(MUTAGDataset):
    """MUTAG dataset. Same as MUTAGDataset.
    """
    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        deprecate_class('MUTAG', 'MUTAGDataset')
        super(MUTAG, self).__init__(print_every,
                                    insert_reverse,
                                    raw_dir,
                                    force_reload,
                                    verbose)

class BGSDataset(RDFGraphDataset):
    """BGS dataset.
    BGS namespace convention:
    http://data.bgs.ac.uk/(ref|id)/<Major Concept>/<Sub Concept>/INSTANCE
    We ignored all literal nodes and the relations connecting them in the
    output graph. We also ignored the relation used to mark whether a
    term is CURRENT or DEPRECATED.
    Statistics
    ===
    Nodes: xxx
    Edges: xxx
    Node Types: xxx
    Edge Types: xxx
    Target Category:
    Number of Classes: xx
    Label Split: Train: xx ,Valid: xx, Test: xx
    Parameters
    -----------
    print_every: int
        Log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.
    Returns
    ===
    BGSDataset object with three properties:
        graph: A Heterogenous graph containing the
            graph structure, node features and labels.
        predict_category: The category name to run the node classification
            prediction.
        num_of_class: number of publication categories
            for the classification task.
    Examples
    --------
    >>> dataset = dgl.data.rdf.BGS()
    >>> print(dataset.graph)
    """

    lith = rdf.term.URIRef("http://data.bgs.ac.uk/ref/Lexicon/hasLithogenesis")
    entity_prefix = 'http://data.bgs.ac.uk/'
    status_prefix = 'http://data.bgs.ac.uk/ref/CurrentStatus'
    relation_prefix = 'http://data.bgs.ac.uk/ref'

    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        url = _get_dgl_url('dataset/rdf/bgs-hetero.zip')
        name = 'bgs-hetero'
        predict_category = 'Lexicon/NamedRockUnit'
        super(BGSDataset, self).__init__(name, url, predict_category,
                                         print_every=print_every,
                                         insert_reverse=insert_reverse,
                                         raw_dir=raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose)

    def parse_entity(self, term):
        if isinstance(term, rdf.Literal):
            return None
        elif isinstance(term, rdf.BNode):
            return None
        entstr = str(term)
        if entstr.startswith(self.status_prefix):
            return None
        if entstr.startswith(self.entity_prefix):
            sp = entstr.split('/')
            if len(sp) != 7:
                return None
            # instance
            cls = '%s/%s' % (sp[4], sp[5])
            inst = sp[6]
            return Entity(id=inst, cls=cls)
        else:
            return None

    def parse_relation(self, term):
        if term == self.lith:
            return None
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            sp = relstr.split('/')
            if len(sp) < 6:
                return None
            assert len(sp) == 6, relstr
            cls = '%s/%s' % (sp[4], sp[5])
            return Relation(cls=cls)
        else:
            relstr = relstr.replace('.', '_')
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        _, rock, label = line.strip().split('\t')
        return rock, label

class BGS(BGSDataset):
    """BGS dataset. Same as BGSDataset.
    """
    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        deprecate_class('BGS', 'BGSDataset')
        super(BGS, self).__init__(print_every,
                                  insert_reverse,
                                  raw_dir,
                                  force_reload,
                                  verbose)


class AMDataset(RDFGraphDataset):
    """AM dataset.
    Namespace convention:
    Instance: http://purl.org/collections/nl/am/<type>-<id>
    Relation: http://purl.org/collections/nl/am/<name>
    We ignored all literal nodes and the relations connecting them in the
    output graph.

    Statistics
    ===
    Nodes: xxx
    Edges: xxx
    Node Types: xxx
    Edge Types: xxx
    Target Category:
    Number of Classes:
    Label Split: Train: xx ,Valid: xx, Test: xx

    Parameters
    -----------
    print_every: int
        Log for every X tuples. Default: 10000.
    insert_reverse: bool
        If true, add reverse edge and reverse relations to the final graph. Default: True.
    raw_dir : str
        Raw file directory to download/contains the input data directory.
        Default: ~/.dgl/
    force_reload : bool
        Whether to reload the dataset. Default: False
    verbose: bool
      Whether to print out progress information. Default: True.

    Returns
    ===
    AMDataset object with three properties:
        graph: A Heterogenous graph containing the
            graph structure, node features and labels.
        predict_category: The category name to run the node classification
            prediction.
        num_of_class: number of publication categories
            for the classification task.

    Examples
    --------
    >>> dataset = dgl.data.rdf.AM()
    >>> print(dataset.graph)
    """

    objectCategory = rdf.term.URIRef("http://purl.org/collections/nl/am/objectCategory")
    material = rdf.term.URIRef("http://purl.org/collections/nl/am/material")
    entity_prefix = 'http://purl.org/collections/nl/am/'
    relation_prefix = entity_prefix

    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        url = _get_dgl_url('dataset/rdf/am-hetero.zip')
        name = 'am-hetero'
        predict_category = 'proxy'
        super(AMDataset, self).__init__(name, url, predict_category,
                                        print_every=print_every,
                                        insert_reverse=insert_reverse,
                                        raw_dir=raw_dir,
                                        force_reload=force_reload,
                                        verbose=verbose)

    def parse_entity(self, term):
        if isinstance(term, rdf.Literal):
            return None
        elif isinstance(term, rdf.BNode):
            return Entity(id=str(term), cls='_BNode')
        entstr = str(term)
        if entstr.startswith(self.entity_prefix):
            sp = entstr.split('/')
            assert len(sp) == 7, entstr
            spp = sp[6].split('-')
            if len(spp) == 2:
                # instance
                cls, inst = spp
            else:
                cls = 'TYPE'
                inst = spp
            return Entity(id=inst, cls=cls)
        else:
            return None

    def parse_relation(self, term):
        if term == self.objectCategory or term == self.material:
            return None
        relstr = str(term)
        if relstr.startswith(self.relation_prefix):
            sp = relstr.split('/')
            assert len(sp) == 7, relstr
            cls = sp[6]
            return Relation(cls=cls)
        else:
            relstr = relstr.replace('.', '_')
            return Relation(cls=relstr)

    def process_tuple(self, raw_tuple, sbj, rel, obj):
        if sbj is None or rel is None or obj is None:
            return None
        return (sbj, rel, obj)

    def process_idx_file_line(self, line):
        proxy, _, label = line.strip().split('\t')
        return proxy, label

class AM(AMDataset):
    """AM dataset. Same as AMDataset.
    """
    def __init__(self,
                 print_every=10000,
                 insert_reverse=True,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True):
        deprecate_class('AM', 'AMDataset')
        super(AM, self).__init__(print_every,
                                 insert_reverse,
                                 raw_dir,
                                 force_reload,
                                 verbose)

if __name__ == '__main__':
    dataset = AIFB()
