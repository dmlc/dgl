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
from .utils import download, extract_archive, get_download_dir, _get_dgl_url

__all__ = ['AIFB', 'MUTAG', 'BGS', 'AM']

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

class RDFGraphDataset:
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
    url : str or path
        URL to download the raw dataset.
    name : str
        Name of the dataset
    force_reload : bool, optional
        If true, force load and process from raw data. Ignore cached pre-processed data.
    print_every : int, optional
        Log for every X tuples.
    insert_reverse : bool, optional
        If true, add reverse edge and reverse relations to the final graph.
    """
    def __init__(self, url, name,
                 force_reload=False,
                 print_every=10000,
                 insert_reverse=True):
        download_dir = get_download_dir()
        zip_file_path = os.path.join(download_dir, '{}.zip'.format(name))
        download(url, path=zip_file_path)
        self._dir = os.path.join(download_dir, name)
        extract_archive(zip_file_path, self._dir)
        self._print_every = print_every
        self._insert_reverse = insert_reverse
        if not force_reload and self.has_cache():
            print('Found cached graph. Load cache ...')
            self.load_cache()
        else:
            raw_tuples = self.load_raw_tuples()
            self.process_raw_tuples(raw_tuples)
        print('#Training samples:', len(self.train_idx))
        print('#Testing samples:', len(self.test_idx))
        print('#Classes:', self.num_classes)
        print('Predict category:', self.predict_category)

    def load_raw_tuples(self):
        raw_rdf_graphs = []
        for i, filename in enumerate(os.listdir(self._dir)):
            fmt = None
            if filename.endswith('nt'):
                fmt = 'nt'
            elif filename.endswith('n3'):
                fmt = 'n3'
            if fmt is None:
                continue
            g = rdf.Graph()
            print('Parsing file %s ...' % filename)
            g.parse(os.path.join(self._dir, filename), format=fmt)
            raw_rdf_graphs.append(g)
        return itertools.chain(*raw_rdf_graphs)

    def process_raw_tuples(self, raw_tuples):
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
            if i % self._print_every == 0:
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
            print('Adding reverse edges ...')
            newsrc = np.hstack([src, dst])
            newdst = np.hstack([dst, src])
            src = newsrc
            dst = newdst
            etid = np.hstack([etid, etid + len(etypes)])
            etypes.extend(['rev-%s' % t for t in etypes])

        self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)

        print('Load training/validation/testing split ...')
        idmap = F.asnumpy(self.graph.nodes[self.predict_category].data[dgl.NID])
        glb2lcl = {glbid : lclid for lclid, glbid in enumerate(idmap)}
        def findidfn(ent):
            if ent not in entities:
                return None
            else:
                return glb2lcl[entities[ent]]
        self.load_data_split(findidfn)

        self.save_cache(mg, src, dst, ntid, etid, ntypes, etypes)

    def build_graph(self, mg, src, dst, ntid, etid, ntypes, etypes):
        # create homo graph
        print('Creating one whole graph ...')
        g = dgl.graph((src, dst))
        g.ndata[dgl.NTYPE] = F.tensor(ntid)
        g.edata[dgl.ETYPE] = F.tensor(etid)
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
        print('Convert to heterograph ...')
        hg = dgl.to_hetero(g,
                           ntypes,
                           etypes,
                           metagraph=mg)
        print('#Node types:', len(hg.ntypes))
        print('#Canonical edge types:', len(hg.etypes))
        print('#Unique edge type names:', len(set(hg.etypes)))
        self.graph = hg

    def save_cache(self, mg, src, dst, ntid, etid, ntypes, etypes):
        nx.write_gpickle(mg, os.path.join(self._dir, 'cached_mg.gpickle'))
        np.save(os.path.join(self._dir, 'cached_src.npy'), src)
        np.save(os.path.join(self._dir, 'cached_dst.npy'), dst)
        np.save(os.path.join(self._dir, 'cached_ntid.npy'), ntid)
        np.save(os.path.join(self._dir, 'cached_etid.npy'), etid)
        save_strlist(os.path.join(self._dir, 'cached_ntypes.txt'), ntypes)
        save_strlist(os.path.join(self._dir, 'cached_etypes.txt'), etypes)
        np.save(os.path.join(self._dir, 'cached_train_idx.npy'), F.asnumpy(self.train_idx))
        np.save(os.path.join(self._dir, 'cached_test_idx.npy'), F.asnumpy(self.test_idx))
        np.save(os.path.join(self._dir, 'cached_labels.npy'), F.asnumpy(self.labels))

    def has_cache(self):
        return (os.path.exists(os.path.join(self._dir, 'cached_mg.gpickle'))
            and os.path.exists(os.path.join(self._dir, 'cached_src.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_dst.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_ntid.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_etid.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_ntypes.txt'))
            and os.path.exists(os.path.join(self._dir, 'cached_etypes.txt'))
            and os.path.exists(os.path.join(self._dir, 'cached_train_idx.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_test_idx.npy'))
            and os.path.exists(os.path.join(self._dir, 'cached_labels.npy')))

    def load_cache(self):
        mg = nx.read_gpickle(os.path.join(self._dir, 'cached_mg.gpickle'))
        src = np.load(os.path.join(self._dir, 'cached_src.npy'))
        dst = np.load(os.path.join(self._dir, 'cached_dst.npy'))
        ntid = np.load(os.path.join(self._dir, 'cached_ntid.npy'))
        etid = np.load(os.path.join(self._dir, 'cached_etid.npy'))
        ntypes = load_strlist(os.path.join(self._dir, 'cached_ntypes.txt'))
        etypes = load_strlist(os.path.join(self._dir, 'cached_etypes.txt'))
        self.train_idx = F.tensor(np.load(os.path.join(self._dir, 'cached_train_idx.npy')))
        self.test_idx = F.tensor(np.load(os.path.join(self._dir, 'cached_test_idx.npy')))
        labels = np.load(os.path.join(self._dir, 'cached_labels.npy'))
        self.num_classes = labels.max() + 1
        self.labels = F.tensor(labels)

        self.build_graph(mg, src, dst, ntid, etid, ntypes, etypes)

    def load_data_split(self, ent2id):
        label_dict = {}
        labels = np.zeros((self.graph.number_of_nodes(self.predict_category),)) - 1
        train_idx = self.parse_idx_file(
            os.path.join(self._dir, 'trainingSet.tsv'),
            ent2id, label_dict, labels)
        test_idx = self.parse_idx_file(
            os.path.join(self._dir, 'testSet.tsv'),
            ent2id, label_dict, labels)
        self.train_idx = F.tensor(train_idx)
        self.test_idx = F.tensor(test_idx)
        self.labels = F.tensor(labels).long()
        self.num_classes = len(label_dict)

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

    @property
    @abc.abstractmethod
    def predict_category(self):
        """Return the category name that has labels."""
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

class AIFB(RDFGraphDataset):
    """AIFB dataset.
    
    Examples
    --------
    >>> dataset = dgl.data.rdf.AIFB()
    >>> print(dataset.graph)
    """

    employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
    affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")
    entity_prefix = 'http://www.aifb.uni-karlsruhe.de/'
    relation_prefix = 'http://swrc.ontoware.org/'

    def __init__(self,
                 force_reload=False,
                 print_every=10000,
                 insert_reverse=True):
        url = _get_dgl_url('dataset/rdf/aifb-hetero.zip')
        name = 'aifb-hetero'
        super(AIFB, self).__init__(url, name,
                                   force_reload=force_reload,
                                   print_every=print_every,
                                   insert_reverse=insert_reverse)

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

    @property
    def predict_category(self):
        return 'Personen'

class MUTAG(RDFGraphDataset):
    """MUTAG dataset.
    
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
                 force_reload=False,
                 print_every=10000,
                 insert_reverse=True):
        url = _get_dgl_url('dataset/rdf/mutag-hetero.zip')
        name = 'mutag-hetero'
        super(MUTAG, self).__init__(url, name,
                                   force_reload=force_reload,
                                   print_every=print_every,
                                   insert_reverse=insert_reverse)

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

    @property
    def predict_category(self):
        return 'd'

class BGS(RDFGraphDataset):
    """BGS dataset.

    BGS namespace convention:
    http://data.bgs.ac.uk/(ref|id)/<Major Concept>/<Sub Concept>/INSTANCE

    We ignored all literal nodes and the relations connecting them in the
    output graph. We also ignored the relation used to mark whether a
    term is CURRENT or DEPRECATED.

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
                 force_reload=False,
                 print_every=10000,
                 insert_reverse=True):
        url = _get_dgl_url('dataset/rdf/bgs-hetero.zip')
        name = 'bgs-hetero'
        super(BGS, self).__init__(url, name,
                                   force_reload=force_reload,
                                   print_every=print_every,
                                   insert_reverse=insert_reverse)

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

    @property
    def predict_category(self):
        return 'Lexicon/NamedRockUnit'

class AM(RDFGraphDataset):
    """AM dataset.

    Namespace convention:
    Instance: http://purl.org/collections/nl/am/<type>-<id>
    Relation: http://purl.org/collections/nl/am/<name>

    We ignored all literal nodes and the relations connecting them in the
    output graph.

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
                 force_reload=False,
                 print_every=10000,
                 insert_reverse=True):
        url = _get_dgl_url('dataset/rdf/am-hetero.zip')
        name = 'am-hetero'
        super(AM, self).__init__(url, name,
                                   force_reload=force_reload,
                                   print_every=print_every,
                                   insert_reverse=insert_reverse)

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

    @property
    def predict_category(self):
        return 'proxy'

if __name__ == '__main__':
    AIFB()
