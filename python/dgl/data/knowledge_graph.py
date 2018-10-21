""" Knowledge graph dataset for rgcn
Code adapted from tkipf/relational-gcn
https://github.com/tkipf/relational-gcn
"""

from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import os, re, sys, gzip
import rdflib as rdf
import pandas as pd
from collections import Counter, defaultdict

from dgl.data.utils import download, extract_archive, get_download_dir

np.random.seed(123)

_urls = {
    'am': ['https://www.dropbox.com/s/t60huxz616x4c4o/am.tgz?dl=1', 'https://www.dropbox.com/s/htisydfgwxmrx65/am_stripped.nt.gz?dl=1'],
    'aifb': ['https://www.dropbox.com/s/0emedu261l4la82/aifb.tgz?dl=1', 'https://www.dropbox.com/s/fkvgvkygo2gf28k/aifb_stripped.nt.gz?dl=1'],
    'bgs': ['https://www.dropbox.com/s/5wzxsuuof185p12/bgs.tgz?dl=1', 'https://www.dropbox.com/s/uqi0k9jd56j02gh/bgs_stripped.nt.gz?dl=1'],
    'mutag': ['https://www.dropbox.com/s/k4y1qpni83dvei1/mutag.tgz?dl=1', 'https://www.dropbox.com/s/qy8j3p8eacvm4ir/mutag_stripped.nt.gz?dl=1'],
    'entity_classify': 'https://www.dropbox.com/s/babuor115oqq2i3/rgcn_entity_classify.tgz?dl=1',
    'FB15k-237': 'https://www.dropbox.com/s/werqxn21mt19nj4/FB15k-237.tgz?dl=1',
    'FB15k': 'https://www.dropbox.com/s/zbyvjuwu1phlxb5/FB15k.tgz?dl=1',
    'wn18': 'https://www.dropbox.com/s/53fvtwxe70j3aon/wn18.tgz?dl=1'
}


class RGCNEntityDataset(object):
    def __init__(self, name):
        self.name = name
        self.dir = get_download_dir()
        tgz_path = os.path.join(self.dir, '{}.tgz'.format(self.name))
        download(_urls[self.name][0], tgz_path)
        extract_archive(tgz_path, self.dir)
        self.dir = os.path.join(self.dir, self.name)
        graph_file_path = os.path.join(self.dir, '{}_stripped.nt.gz'.format(self.name))
        download(_urls[self.name][1], path=graph_file_path) # no need to uncompress

    def load(self, bfs_level=2, relabel=False):
        self.num_nodes, edges, self.num_rels, self.labels, labeled_nodes_idx, self.train_idx, self.test_idx = _load_data(self.name, self.dir)

        # bfs to reduce edges
        if bfs_level > 0:
            print("removing nodes that are more than {} hops away".format(bfs_level))
            row, col, edge_type = edges.transpose()
            A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(self.num_nodes, self.num_nodes))
            bfs_generator = _bfs_relational(A, labeled_nodes_idx)
            lvls = list()
            lvls.append(set(labeled_nodes_idx))
            for _ in range(bfs_level):
                lvls.append(next(bfs_generator))
            to_delete = list(set(range(self.num_nodes)) - set.union(*lvls))
            eid_to_delete = np.isin(row, to_delete) + np.isin(col, to_delete)
            eid_to_keep = np.logical_not(eid_to_delete)
            self.edge_src = row[eid_to_keep]
            self.edge_dst = col[eid_to_keep]
            self.edge_type = edge_type[eid_to_keep]

            if relabel:
                uniq_nodes, edges = np.unique((self.edge_src, self.edge_dst), return_inverse=True)
                self.src, self.dst = np.reshape(edges, (-1, 2))
                node_map = np.zeros(self.num_nodes, dtype=int)
                self.num_nodes = len(uniq_nodes)
                node_map[uniq_nodes] = np.arange(self.num_nodes)
                self.labels = self.labels[uniq_nodes]
                self.train_idx = node_map[self.train_idx]
                self.test_idx = node_map[self.test_idx]
                print("{} nodes left".format(self.num_nodes))
        else:
            self.src, self.dst, self.edge_type = edges.transpose()

        # normalize by dst degree
        _, inverse_index, count = np.unique((self.dst, self.edge_type), return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        self.edge_norm = np.ones(num_edges, dtype=np.float) / degrees


class RGCNLinkDataset(object):
    def __init__(self, name):
        self.name = name
        self.dir = get_download_dir()
        tgz_file_path = os.path.join(self.dir, '{}.tar.gz'.format(self.name))
        download(_urls[self.name], tgz_file_path)
        extract_archive(tgz_file_path, self.dir)
        self.dir = os.path.join(self.dir, self.name)

    def load(self):
        entity_path = os.path.join(self.dir, 'entities.dict')
        relation_path = os.path.join(self.dir, 'relations.dict')
        train_path = os.path.join(self.dir, 'train.txt')
        valid_path = os.path.join(self.dir, 'valid.txt')
        test_path = os.path.join(self.dir, 'test.txt')
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        self.train = _read_triplets_as_list(train_path, entity_dict, relation_dict)
        self.valid = _read_triplets_as_list(valid_path, entity_dict, relation_dict)
        self.test = _read_triplets_as_list(test_path, entity_dict, relation_dict)
        self.num_nodes = len(entity_dict)
        num_rels = len(relation_dict)
        print("# entities: {}".format(self.num_nodes))
        print("# relations: {}".format(num_rels))
        self.edges, self.relations = self.build_adj(self.train, num_rels)
        print("# edges: {}".format(len(self.relations)))

    def build_adj(self, triplets, num_rels):
        num_rels *= 2 # add reverse
        edge_dict = defaultdict(lambda: np.zeros(num_rels, dtype=np.float32))
        for s, r, o in triplets:
            edge_dict[(s, o)][2 * r] = 1
            edge_dict[(o, s)][2 * r + 1] = 1 # reverse rel

        edges = sorted(list(edge_dict.keys()))
        edge_types = np.stack([edge_dict[e] for e in edges])
        edges = np.array(edges)

        # normalize edge_types for edge repr
        # count edge type for each edge
        edge_type_count = np.sum(edge_types, axis=1)
        src, dst = edges.transpose()
        coo = sp.coo_matrix((edge_type_count, (src, dst)), shape=[self.num_nodes, self.num_nodes])
        # normalize per dst node
        edge_count = np.reshape(np.asarray(coo.sum(axis=0)), (-1, 1))
        edge_values = edge_types / edge_count[dst]
        return edges, edge_values


def load_entity(args):
    data = RGCNEntityDataset(args.dataset)
    data.load(args.bfs_level, args.relabel)
    return data

def load_link(args):
    data = RGCNLinkDataset(args.dataset)
    data.load()
    return data


def _sp_row_vec_from_idx_list(idx_list, dim):
    """Create sparse vector of dimensionality dim from a list of indices."""
    shape = (1, dim)
    data = np.ones(len(idx_list))
    row_ind = np.zeros(len(idx_list))
    col_ind = list(idx_list)
    return sp.csr_matrix((data, (row_ind, col_ind)), shape=shape)

def _get_neighbors(adj, nodes):
    """Takes a set of nodes and a graph adjacency matrix and returns a set of neighbors."""
    sp_nodes = _sp_row_vec_from_idx_list(list(nodes), adj.shape[1])
    sp_neighbors = sp_nodes.dot(adj)
    neighbors = set(sp.find(sp_neighbors)[1])  # convert to set of indices
    return neighbors

def _bfs_relational(adj, roots):
    """
    BFS for graphs with multiple edge types. Returns list of level sets.
    Each entry in list corresponds to relation specified by adj_list.
    """
    visited = set()
    current_lvl = set(roots)

    next_lvl = set()

    while current_lvl:

        for v in current_lvl:
            visited.add(v)

        next_lvl = _get_neighbors(adj, current_lvl)
        next_lvl -= visited  # set difference

        yield next_lvl

        current_lvl = set.union(next_lvl)


class RDFReader(object):
    __graph = None
    __freq = {}

    def __init__(self, file):

        self.__graph = rdf.Graph()

        if file.endswith('nt.gz'):
            with gzip.open(file, 'rb') as f:
                self.__graph.parse(file=f, format='nt')
        else:
            self.__graph.parse(file, format=rdf.util.guess_format(file))

        # See http://rdflib.readthedocs.io for the rdflib documentation

        self.__freq = Counter(self.__graph.predicates())

        print("Graph loaded, frequencies counted.")

    def triples(self, relation=None):
        for s, p, o in self.__graph.triples((None, relation, None)):
            yield s, p, o

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.__graph.destroy("store")
        self.__graph.close(True)

    def subjectSet(self):
        return set(self.__graph.subjects())

    def objectSet(self):
        return set(self.__graph.objects())

    def relationList(self):
        """
        Returns a list of relations, ordered descending by frequenecy
        :return:
        """
        res = list(set(self.__graph.predicates()))
        res.sort(key=lambda rel: - self.freq(rel))
        return res

    def __len__(self):
        return len(self.__graph)

    def freq(self, rel):
        if rel not in self.__freq:
            return 0
        return self.__freq[rel]


def _load_sparse_csr(filename):
    loader = np.load(filename)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'], dtype=np.float32)


def _save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)


def _load_data(dataset_str='aifb', dataset_path=None):
    """

    :param dataset_str:
    :param rel_layers:
    :param limit: If > 0, will only load this many adj. matrices
        All adjacencies are preloaded and saved to disk,
        but only a limited a then restored to memory.
    :return:
    """

    print('Loading dataset', dataset_str)

    dirname = os.path.dirname(os.path.realpath(sys.argv[0]))

    graph_file = os.path.join(dataset_path, '{}_stripped.nt.gz'.format(dataset_str))
    task_file = os.path.join(dataset_path, 'completeDataset.tsv')
    train_file = os.path.join(dataset_path, 'trainingSet.tsv')
    test_file = os.path.join(dataset_path, 'testSet.tsv')
    if dataset_str == 'am':
        label_header = 'label_cateogory'
        nodes_header = 'proxy'
    elif dataset_str == 'aifb':
        label_header = 'label_affiliation'
        nodes_header = 'person'
    elif dataset_str == 'mutag':
        label_header = 'label_mutagenic'
        nodes_header = 'bond'
    elif dataset_str == 'bgs':
        label_header = 'label_lithogenesis'
        nodes_header = 'rock'
    else:
        raise NameError('Dataset name not recognized: ' + dataset_str)

    edge_file = os.path.join(dataset_path, 'edges.npz')
    labels_file = os.path.join(dataset_path, 'labels.npz')
    train_idx_file = os.path.join(dataset_path, 'train_idx.npy')
    test_idx_file = os.path.join(dataset_path, 'test_idx.npy')
    # train_names_file = os.path.join(dataset_path, 'train_names.npy')
    # test_names_file = os.path.join(dataset_path, 'test_names.npy')
    # rel_dict_file = os.path.join(dataset_path, 'rel_dict.pkl')
    # nodes_file = os.path.join(dataset_path, 'nodes.pkl')

    if os.path.isfile(adj_file) and os.path.isfile(labels_file) and \
            os.path.isfile(train_idx_file) and os.path.isfile(test_idx_file):

        # load precomputed adjacency matrix and labels
        all_edges = np.load(edge_file)
        num_node = all_edges['n'].item()
        edge_list = all_edges['edges']
        num_rel = all_edges['nrel']

        print('Number of nodes: ', num_node)
        print('Number of relations: ', num_rel)

        labels = _load_sparse_csr(labels_file)
        labeled_nodes_idx = list(labels.nonzero()[0])

        print('Number of classes: ', labels.shape[1])

        train_idx = np.load(train_idx_file)
        test_idx = np.load(test_idx_file)

        # train_names = np.load(train_names_file)
        # test_names = np.load(test_names_file)
        # relations_dict = pkl.load(open(rel_dict_file, 'rb'))

    else:

        # loading labels of nodes
        labels_df = pd.read_csv(task_file, sep='\t', encoding='utf-8')
        labels_train_df = pd.read_csv(train_file, sep='\t', encoding='utf8')
        labels_test_df = pd.read_csv(test_file, sep='\t', encoding='utf8')

        with RDFReader(graph_file) as reader:

            relations = reader.relationList()
            subjects = reader.subjectSet()
            objects = reader.objectSet()

            nodes = list(subjects.union(objects))
            num_node = len(nodes)
            num_rel = len(relations)
            num_rel = 2 * num_rel + 1 # +1 is for self-relation

            assert num_node < np.iinfo(np.int32).max
            print('Number of nodes: ', num_node)
            print('Number of relations: ', num_rel)

            edge_list = []
            relations_dict = {rel: i for i, rel in enumerate(list(relations))}
            nodes_dict = {node: i for i, node in enumerate(nodes)}

            # self relation
            for i in range(num_node):
                edge_list.append(i, i, 0)

            for i, (s, p, o) in enumerate(reader.triples()):
                src = node_dict[s]
                dst = node_dict[o]
                assert src < num_node and dst < num_node
                rel = relations_dict[p]
                edge_list.append(src, dst, 2 * rel + 1)
                edge_list.append(dst, src, 2 * rel + 1)

            # sort indices by destination
            edge_list = sorted(edge_list, key=lambda x: (x[1], x[0], x[2]))
            edge_list = np.array(edge_list, dtype=np.int)

            edge_types = []
            for edge in all_edges:
                edge_types.append(edge_dict[edge])
            all_edges = np.array(all_edges, dtype=np.int)
            np.savez(adj_file, edges=edge_list, n=np.array(num_node), nrel=np.array(num_rel))

        # Reload the adjacency matrices from disk
        adjmat = np.load(adj_file)

        nodes_u_dict = {np.unicode(to_unicode(key)): val for key, val in
                        nodes_dict.items()}

        labels_set = set(labels_df[label_header].values.tolist())
        labels_dict = {lab: i for i, lab in enumerate(list(labels_set))}

        print('{} classes: {}'.format(len(labels_set), labels_set))

        labels = sp.lil_matrix((num_node, len(labels_set)))
        labeled_nodes_idx = []

        print('Loading training set')

        train_idx = []
        train_names = []
        for nod, lab in zip(labels_train_df[nodes_header].values,
                            labels_train_df[label_header].values):
            nod = np.unicode(to_unicode(nod))  # type: unicode
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                train_idx.append(nodes_u_dict[nod])
                train_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        print('Loading test set')

        test_idx = []
        test_names = []
        for nod, lab in zip(labels_test_df[nodes_header].values,
                            labels_test_df[label_header].values):
            nod = np.unicode(to_unicode(nod))
            if nod in nodes_u_dict:
                labeled_nodes_idx.append(nodes_u_dict[nod])
                label_idx = labels_dict[lab]
                labels[labeled_nodes_idx[-1], label_idx] = 1
                test_idx.append(nodes_u_dict[nod])
                test_names.append(nod)
            else:
                print(u'Node not in dictionary, skipped: ',
                      nod.encode('utf-8', errors='replace'))

        labeled_nodes_idx = sorted(labeled_nodes_idx)
        labels = labels.tocsr()
        print('Number of classes: ', labels.shape[1])

        _save_sparse_csr(labels_file, labels)

        np.save(train_idx_file, train_idx)
        np.save(test_idx_file, test_idx)

        # np.save(train_names_file, train_names)
        # np.save(test_names_file, test_names)

        # pkl.dump(relations_dict, open(rel_dict_file, 'wb'))

    # end if

    return num_node, edge_list, num_rel, labels, labeled_nodes_idx, train_idx, test_idx


def to_unicode(input):
    # FIXME (lingfan): not sure about python 2 and 3 str compatibility
    return str(input)
    """ lingfan: comment out for now
    if isinstance(input, unicode):
        return input
    elif isinstance(input, str):
        return input.decode('utf-8', errors='replace')
    return str(input).decode('utf-8', errors='replace')
    """


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
