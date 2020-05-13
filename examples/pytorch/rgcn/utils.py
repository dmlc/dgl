"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import os
import traceback
from _thread import start_new_thread
from functools import wraps

import numpy as np
import pandas as pd
import torch
from torch.multiprocessing import Queue
import dgl

from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = edges.transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    print("# sampled nodes: {}".format(len(uniq_v)))
    print("# sampled edges: {}".format(len(src) * 2))
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    print("# nodes: {}, # edges: {}".format(num_nodes, len(src)))
    return g, rel.astype('int64'), norm.astype('int64')

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = edges.transpose()
    print("Test graph:")
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility functions for evaluations (raw)
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices

def perturb_and_get_raw_rank(embedding, w, a, r, b, test_size, batch_size=100):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    for idx in range(n_batch):
        print("batch {} / {}".format(idx, n_batch))
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))
    return torch.cat(ranks)

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_raw_mrr(embedding, w, test_triplets, hits=[], eval_bz=100):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_raw_rank(embedding, w, o, r, s, test_size, eval_bz)
        # perturb object
        ranks_o = perturb_and_get_raw_rank(embedding, w, s, r, o, test_size, eval_bz)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Utility functions for evaluations (filtered)
#
#######################################################################

def filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_o = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider an object if it is part of a triplet to filter
    for o in range(num_entities):
        if (target_s, target_r, o) not in triplets_to_filter:
            filtered_o.append(o)
    return torch.LongTensor(filtered_o)

def filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities):
    target_s, target_r, target_o = int(target_s), int(target_r), int(target_o)
    filtered_s = []
    # Do not filter out the test triplet, since we want to predict on it
    if (target_s, target_r, target_o) in triplets_to_filter:
        triplets_to_filter.remove((target_s, target_r, target_o))
    # Do not consider a subject if it is part of a triplet to filter
    for s in range(num_entities):
        if (s, target_r, target_o) not in triplets_to_filter:
            filtered_s.append(s)
    return torch.LongTensor(filtered_s)

def perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb object in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_o = filter_o(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_o_idx = int((filtered_o == target_o).nonzero())
        emb_s = embedding[target_s]
        emb_r = w[target_r]
        emb_o = embedding[filtered_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_o_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter):
    """ Perturb subject in the triplets
    """
    num_entities = embedding.shape[0]
    ranks = []
    for idx in range(test_size):
        if idx % 100 == 0:
            print("test triplet {} / {}".format(idx, test_size))
        target_s = s[idx]
        target_r = r[idx]
        target_o = o[idx]
        filtered_s = filter_s(triplets_to_filter, target_s, target_r, target_o, num_entities)
        target_s_idx = int((filtered_s == target_s).nonzero())
        emb_s = embedding[filtered_s]
        emb_r = w[target_r]
        emb_o = embedding[target_o]
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        rank = int((indices == target_s_idx).nonzero())
        ranks.append(rank)
    return torch.LongTensor(ranks)

def calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[]):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        triplets_to_filter = torch.cat([train_triplets, valid_triplets, test_triplets]).tolist()
        triplets_to_filter = {tuple(triplet) for triplet in triplets_to_filter}
        print('Perturbing subject...')
        ranks_s = perturb_s_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)
        print('Perturbing object...')
        ranks_o = perturb_o_and_get_filtered_rank(embedding, w, s, r, o, test_size, triplets_to_filter)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (filtered): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (filtered) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()

#######################################################################
#
# Main evaluation function
#
#######################################################################

def calc_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits=[], eval_bz=100, eval_p="filtered"):
    if eval_p == "filtered":
        mrr = calc_filtered_mrr(embedding, w, train_triplets, valid_triplets, test_triplets, hits)
    else:
        mrr = calc_raw_mrr(embedding, w, test_triplets, hits, eval_bz)
    return mrr
#######################################################################
#
# DRKG Dataset
#
#######################################################################
class DrkgDataset(object):
    def __init__(self, name='drkg'):
        self.name = name
        self.dir = get_download_dir()
        _downlaod_prefix = 'https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/'
        tgz_path = os.path.join(self.dir, '{}.tar.gz'.format(self.name))
        download(_downlaod_prefix + '{}.tar.gz'.format(self.name), tgz_path)
        self.dir = os.path.join(self.dir, self.name)
        extract_archive(tgz_path, self.dir)

    def load_data(self):
        raw_file = os.path.join(self.dir, 'drkg/drkg.tsv')
        drkg_triplets = pd.read_csv(raw_file, sep='\t').values
        np.random.seed(42) # ensure each time the split is same
        num_of_triplets = len(drkg_triplets)
        train_ratio = 0.9
        valid_ratio = 0.05
        test_ratio = 0.05
        train_cnt = int(num_of_triplets * train_ratio)
        valid_cnt = int(num_of_triplets * valid_ratio)
        test_cnt = num_of_triplets - train_cnt - valid_cnt

        seeds = np.arange(num_of_triplets)
        np.random.shuffle(seeds)

        train_triplets = drkg_triplets[seeds[:train_cnt]]
        valid_triplets = drkg_triplets[seeds[train_cnt:train_cnt+valid_cnt]]
        test_triplets = drkg_triplets[seeds[train_cnt+valid_cnt:]]

        def handle_node(node):
            node_info = node.split('::')

            # node_type, node_name
            if len(node_info) == 1:
                return None, None
            return node_info[0], node_info[1]

        def handle_relation(relation):
            return relation

        def get_id(dict, key):
            id = dict.get(key, None)
            if id is None:
                id = len(dict)
                dict[key] = id
            return id

        entity_type_dict = {}
        entity_dict = {}
        relation_dict = {}
        node_types = []
        graph_relations = set()
        def handle_triples(triplets):
            heads = []
            head_types = []
            rel_types = []
            tails = []
            tail_types = []
            for triplet in triplets:
                head_type, head_name = handle_node(triplet[0])
                if head_type is None:
                    continue
                tail_type, tail_name = handle_node(triplet[2])
                if tail_type is None:
                    continue
                # here the rel type is already (h, r, t)
                rel_type = (head_type, triplet[1], tail_type)

                rel_id = get_id(relation_dict, rel_type)
                ed_len = len(entity_dict)
                head_id = get_id(entity_dict, triplet[0])
                head_type_id = get_id(entity_type_dict, head_type)
                if len(entity_dict) > ed_len:
                    node_types.append(head_type_id)
                ed_len = len(entity_dict)
                tail_id = get_id(entity_dict, triplet[2])
                tail_type_id = get_id(entity_type_dict, tail_type)
                if len(entity_dict) > ed_len:
                    node_types.append(tail_type_id)

                heads.append(head_id)
                head_types.append(head_type_id)
                rel_types.append(rel_id)
                tails.append(tail_id)
                tail_types.append(tail_type_id)
                graph_relations.add((head_type_id, rel_id, tail_type_id))

            heads = np.asarray(heads)
            tails = np.asarray(tails)
            head_types = np.asarray(head_types)
            rel_types = np.asarray(rel_types)
            tail_types = np.asarray(tail_types)
            return heads, tails, head_types, rel_types, tail_types

        train_head, train_tail, train_head_type, train_rel_type, train_tail_type = \
            handle_triples(train_triplets)

        valid_head, valid_tail, valid_head_type, valid_rel_type, valid_tail_type = \
            handle_triples(valid_triplets)

        test_head, test_tail, test_head_type, test_rel_type, test_tail_type = \
            handle_triples(test_triplets)

        self.train = (train_head, train_tail, train_head_type, train_rel_type, train_tail_type)
        self.valid = (valid_head, valid_tail, valid_head_type, valid_rel_type, valid_tail_type)
        self.test = (test_head, test_tail, test_head_type, test_rel_type, test_tail_type)
        self.rel_map = relation_dict
        self.entity_map = entity_dict
        self.entity_type_dict = entity_type_dict
        self.num_rels = len(list(graph_relations))
        self.num_nodes = len(entity_dict)
        self.node_types = np.asarray(node_types)

def build_multi_ntype_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, node_types, edge_lists, reverse=True):
    """ Create a DGL homogeneous graph with heterograph info stored as node or edge features.
    """
    src = []
    rel = []
    dst = []
    raw_subg = {}
    raw_subg_eset = {}
    raw_subg_etype = {}
    raw_reverse_sugb = {}
    raw_reverse_subg_etype = {}
    print(num_rels)

    settype = 0
    for edge_list in edge_lists:
        s_l, d_l, st_l, r_l, dt_l = edge_list
        for i in range(s_l.shape[0]):
            s = s_l[i]
            d = d_l[i]
            st = st_l[i]
            r = r_l[i]
            dt = dt_l[i]
            assert r < num_rels
            s_type = str(st)
            d_type = str(dt)
            r_type = str(r)
            e_type = (s_type, r_type, d_type)

            if raw_subg.get(e_type, None) is None:
                raw_subg[e_type] = ([], [])
                raw_subg_eset[e_type] = []
                raw_subg_etype[e_type] = []
            raw_subg[e_type][0].append(s)
            raw_subg[e_type][1].append(d)
            raw_subg_eset[e_type].append(settype)
            raw_subg_etype[e_type].append(r)

            if reverse is True:
                r_type = str(r + num_rels)
                re_type = (d_type, r_type, s_type)
                if raw_reverse_sugb.get(re_type, None) is None:
                    raw_reverse_sugb[re_type] = ([], [])
                    raw_reverse_subg_etype[re_type] = []
                raw_reverse_sugb[re_type][0].append(d)
                raw_reverse_sugb[re_type][1].append(s)
                raw_reverse_subg_etype[re_type].append(r + num_rels)
        settype += 1

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
        etype = torch.tensor(etype).long()
        settype = raw_subg_eset[e_type]
        settype = torch.tensor(settype).long()

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    for e_type, val in raw_reverse_sugb.items():
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)
        etype = raw_reverse_subg_etype[e_type]
        etype = torch.tensor(etype).long()
        settype = torch.full((s.shape[0],), -1).long()

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    s = np.concatenate(fg_s)
    d = np.concatenate(fg_d)
    g = dgl.graph((s, d), num_nodes=num_nodes)
    g.edata['etype'] = torch.cat(fg_etype)
    g.edata['set'] = torch.cat(fg_settype)
    g.ndata['ntype'] = torch.from_numpy(node_types).long()

    return g

#######################################################################
#
# Build Heterograph in Homogeneous Graph for Link Prediction
#
#######################################################################
def build_heterograph_in_homogeneous_from_triplets(num_nodes, num_rels, edge_lists, reverse=True):
    """ Create a DGL homogeneous graph with heterograph info stored as node or edge features.
    """
    src = []
    rel = []
    dst = []
    raw_subg = {}
    raw_subg_eset = {}
    raw_subg_etype = {}
    raw_reverse_sugb = {}
    raw_reverse_subg_etype = {}
    print(num_rels)

    # here there is noly one node type
    s_type = "node"
    d_type = "node"

    setype = 0
    for edge_list in edge_lists:
        for edge in edge_list:
            s, r, d = edge
            assert r < num_rels
            r_type = str(r)
            e_type = (s_type, r_type, d_type)

            if raw_subg.get(e_type, None) is None:
                raw_subg[e_type] = ([], [])
                raw_subg_eset[e_type] = []
                raw_subg_etype[e_type] = []
            raw_subg[e_type][0].append(s)
            raw_subg[e_type][1].append(d)
            raw_subg_eset[e_type].append(setype)
            raw_subg_etype[e_type].append(r)

            if reverse is True:
                r_type = str(r + num_rels)
                re_type = (d_type, r_type, s_type)
                if raw_reverse_sugb.get(re_type, None) is None:
                    raw_reverse_sugb[re_type] = ([], [])
                    raw_reverse_subg_etype[re_type] = []
                raw_reverse_sugb[re_type][0].append(d)
                raw_reverse_sugb[re_type][1].append(s)
                raw_reverse_subg_etype[re_type].append(r + num_rels)
        setype += 1

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
        etype = torch.tensor(etype).long()
        settype = raw_subg_eset[e_type]
        settype = torch.tensor(settype).long()

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    for e_type, val in raw_reverse_sugb.items():
        s, d = val
        s = np.asarray(s)
        d = np.asarray(d)
        etype = raw_reverse_subg_etype[e_type]
        etype = torch.tensor(etype).long()
        settype = torch.full((s.shape[0],), -1).long()

        fg_s.append(s)
        fg_d.append(d)
        fg_etype.append(etype)
        fg_settype.append(settype)

    s = np.concatenate(fg_s)
    d = np.concatenate(fg_d)
    g = dgl.graph((s, d), num_nodes=num_nodes)
    g.edata['etype'] = torch.cat(fg_etype)
    g.edata['set'] = torch.cat(fg_settype)
    g.ndata['ntype'] = torch.full((num_nodes,), 0)
    return g

# According to https://github.com/pytorch/pytorch/issues/17199, this decorator
# is necessary to make fork() and openmp work together.
def thread_wrapped_func(func):
    """
    Wraps a process entry point to make it work with OpenMP.
    """
    @wraps(func)
    def decorated_function(*args, **kwargs):
        queue = Queue()
        def _queue_result():
            exception, trace, res = None, None, None
            try:
                res = func(*args, **kwargs)
            except Exception as e:
                exception = e
                trace = traceback.format_exc()
            queue.put((res, exception, trace))

        start_new_thread(_queue_result, ())
        result, exception, trace = queue.get()
        if exception is None:
            return result
        else:
            assert isinstance(exception, Exception)
            raise exception.__class__(trace)
    return decorated_function