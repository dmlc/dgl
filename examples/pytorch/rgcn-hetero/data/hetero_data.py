from collections import namedtuple, OrderedDict
import os, sys
import rdflib as rdf
import networkx as nx
import numpy as np
import dgl
import torch

employs = rdf.term.URIRef("http://swrc.ontoware.org/ontology#employs")
affiliation = rdf.term.URIRef("http://swrc.ontoware.org/ontology#affiliation")

Entity = namedtuple('Entity', ['id', 'cls', 'attrs'])
Relation = namedtuple('Relation', ['cls', 'attrs'])

class RDFParser:
    def __init__(self):
        self._entity_prefix = 'http://www.aifb.uni-karlsruhe.de/'
        self._relation_prefix = 'http://swrc.ontoware.org/'

    def parse_subject(self, term):
        if isinstance(term, rdf.BNode):
            #return Entity(cls=None, attrs={'data' : str(term)})
            return None
        entstr = str(term)
        if entstr.startswith(self._entity_prefix):
            sp = entstr.split('/')
            return Entity(id=sp[5], cls=sp[3], attrs=None)
        else:
            return None

    def parse_object(self, term):
        if isinstance(term, rdf.Literal):
            #return Entity(id=str(term), cls='_Literal', attrs=None)
            return None
        elif isinstance(term, rdf.BNode):
            #return Entity(cls=None, attrs={'data' : str(term)})
            return None
        entstr = str(term)
        if entstr.startswith(self._entity_prefix):
            sp = entstr.split('/')
            return Entity(id=sp[5], cls=sp[3], attrs=None)
        else:
            return None

    def parse_predicate(self, term):
        if term == employs or term == affiliation:
            return None
        relstr = str(term)
        if relstr.startswith(self._relation_prefix):
            return Relation(cls=relstr.split('/')[3], attrs=None)
        else:
            return Relation(cls=relstr, attrs=None)

def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id

def parse_rdf(g, parser, category, training_set, testing_set, insert_reverse=True):
    mg = nx.MultiDiGraph()
    ent_classes = OrderedDict()
    rel_classes = OrderedDict()
    entities = OrderedDict()
    src = []
    dst = []
    ntid = []
    etid = []
    for i, (sbj, pred, obj) in enumerate(g):
        sbjent = parser.parse_subject(sbj)
        rel = parser.parse_predicate(pred)
        objent = parser.parse_object(obj)
        if sbjent is None or rel is None or objent is None:
            continue
        # meta graph
        sbjclsid = _get_id(ent_classes, sbjent.cls)
        objclsid = _get_id(ent_classes, objent.cls)
        relclsid = _get_id(rel_classes, rel.cls)
        mg.add_edge(sbjent.cls, objent.cls, key=rel.cls)
        if insert_reverse:
            mg.add_edge(objent.cls, sbjent.cls, key='rev-%s' % rel.cls)
        # instance graph
        src_id = _get_id(entities, '%s/%s' % (sbjent.cls, sbjent.id))
        if len(entities) > len(ntid):  # found new entity
            ntid.append(sbjclsid)
        dst_id = _get_id(entities, '%s/%s' % (objent.cls, objent.id))
        if len(entities) > len(ntid):  # found new entity
            ntid.append(objclsid)
        src.append(src_id)
        dst.append(dst_id)
        etid.append(relclsid)
    #print(len(ent_classes))
    #print(ent_classes.keys())
    #print(len(rel_classes))
    #print(rel_classes.keys())
    src = np.array(src)
    dst = np.array(dst)
    ntid = np.array(ntid)
    etid = np.array(etid)
    ntypes = list(ent_classes.keys())
    etypes = list(rel_classes.keys())

    # add reverse edge with reverse relation
    if insert_reverse:
        newsrc = np.hstack([src, dst])
        newdst = np.hstack([dst, src])
        src = newsrc
        dst = newdst
        etid = np.hstack([etid, etid + len(etypes)])
        etypes.extend(['rev-%s' % t for t in etypes])

    # create homo graph
    g = dgl.graph((src, dst), card=len(entities))
    g.ndata[dgl.NTYPE] = torch.tensor(ntid)
    g.edata[dgl.ETYPE] = torch.tensor(etid)
    train_label = torch.zeros((len(entities),), dtype=torch.int64) - 1
    test_label = torch.zeros((len(entities),), dtype=torch.int64) - 1
    for sample, lbl in training_set.items():
        if sample in entities:
            entid = entities[sample]
            train_label[entid] = lbl
        else:
            print('Warning: sample "%s" does not have any valid links associated. Ignored.' % sample)
    for sample, lbl in testing_set.items():
        if sample in entities:
            entid = entities[sample]
            test_label[entid] = lbl
        else:
            print('Warning: sample "%s" does not have any valid links associated. Ignored.' % sample)
    g.ndata['train_label'] = train_label
    g.ndata['test_label'] = test_label
    print('Total #nodes:', g.number_of_nodes())
    print('Total #edges:', g.number_of_edges())

    # convert to heterograph
    hg = dgl.to_hetero(g,
                       ntypes,
                       etypes,
                       metagraph=mg)
    print('#Node types:', len(hg.ntypes))
    print('#Canonical edge types:', len(hg.etypes))
    print('#Unique edge type names:', len(set(hg.etypes)))
    #print(hg.canonical_etypes)
    nx.drawing.nx_pydot.write_dot(mg, 'meta.dot')

    # make training and testing index
    train_label = hg.nodes[category].data['train_label']
    train_idx = torch.nonzero(train_label != -1).squeeze()
    test_label = hg.nodes[category].data['test_label']
    test_idx = torch.nonzero(test_label != -1).squeeze()
    labels = torch.zeros_like(train_label) - 1
    labels[train_idx] = train_label[train_idx]
    labels[test_idx] = test_label[test_idx]
    print('#Training samples:', len(train_idx))
    print('#Testing samples:', len(test_idx))
    return hg, train_idx, test_idx, labels

def parse_idx_file(filename):
    labels = {}
    person2affil = {}
    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # first line is the header
            person, _, label = line.strip().split('\t')
            sp = person.split('/')
            category = sp[3]
            pid = '%s/%s' % (sp[3], sp[5])
            person2affil[pid] = _get_id(labels, label)
    return person2affil, category, len(labels)

def load_hetero():
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'aifb-hetero')
    training_set, category, num_classes = parse_idx_file(os.path.join(dir_path, 'trainingSet.tsv'))
    testing_set, _, num_classes = parse_idx_file(os.path.join(dir_path, 'testSet.tsv'))
    g = rdf.Graph()
    n3path = os.path.join(dir_path, 'aifbfixed_complete.n3')
    g.parse(n3path, format='n3')
    parser = RDFParser()
    hg, train_idx, test_idx, labels = parse_rdf(
            g, parser, category, training_set, testing_set)
    g.close()
    return hg, category, num_classes, train_idx, test_idx, labels
