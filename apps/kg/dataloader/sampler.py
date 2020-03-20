# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time

from dgl.base import NID, EID

def SoftRelationPartition(edges, n, threshold=0.05):
    """This partitions a list of edges to n partitions according to their
    relation types. For any relation with number of edges larger than the
    threshold, its edges will be evenly distributed into all partitions.
    For any relation with number of edges smaller than the threshold, its
    edges will be put into one single partition.

    Algo:
    For r in relations:
        if r.size() > threadold
            Evenly divide edges of r into n parts and put into each relation.
        else
            Find partition with fewest edges, and put edges of r into 
            this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        Number of partitions
    threshold : float
        The threshold of whether a relation is LARGE or SMALL
        Default: 5%

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    cross_rel_part = []
    for _ in range(n):
        rel_parts.append([])

    large_threshold = int(len(rels) * threshold)
    capacity_per_partition = int(len(rels) / n)
    # ensure any relation larger than the partition capacity will be split
    large_threshold = capacity_per_partition if capacity_per_partition < large_threshold \
                      else large_threshold
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []
        if cnt > large_threshold:
            avg_part_cnt = (cnt // n) + 1
            num_cross_part += 1
            for j in range(n):
                part_cnt = avg_part_cnt if cnt > avg_part_cnt else cnt
                r_parts.append([j, part_cnt])
                rel_parts[j].append(r)
                edge_cnts[j] += part_cnt
                rel_cnts[j] += 1
                cnt -= part_cnt
            cross_rel_part.append(r)
        else:
            idx = np.argmin(edge_cnts)
            r_parts.append([idx, cnt])
            rel_parts[idx].append(r)
            edge_cnts[idx] += cnt
            rel_cnts[idx] += 1
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)
    cross_rel_part = np.array(cross_rel_part)

    return parts, rel_parts, num_cross_part > 0, cross_rel_part

def BalancedRelationPartition(edges, n):
    """This partitions a list of edges based on relations to make sure
    each partition has roughly the same number of edges and relations.
    Algo:
    For r in relations:
      Find partition with fewest edges
      if r.size() > num_of empty_slot
         put edges of r into this partition to fill the partition,
         find next partition with fewest edges to put r in.
      else
         put edges of r into this partition.

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    List of np.array
        Edge types of each partition
    bool
        Whether there exists some relations belongs to multiple partitions
    """
    heads, rels, tails = edges
    print('relation partition {} edges into {} parts'.format(len(heads), n))
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    for _ in range(n):
        rel_parts.append([])

    max_edges = (len(rels) // n) + 1
    num_cross_part = 0
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        r_parts = []

        while cnt > 0:
            idx = np.argmin(edge_cnts)
            if edge_cnts[idx] + cnt <= max_edges:
                r_parts.append([idx, cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cnt
                rel_cnts[idx] += 1
                cnt = 0
            else:
                cur_cnt = max_edges - edge_cnts[idx]
                r_parts.append([idx, cur_cnt])
                rel_parts[idx].append(r)
                edge_cnts[idx] += cur_cnt
                rel_cnts[idx] += 1
                num_cross_part += 1
                cnt -= cur_cnt
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_cross_part, len(cnts)))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])

    for i, r in enumerate(rels):
        r_part = rel_dict[r][0]
        part_idx = r_part[0]
        cnt = r_part[1]
        parts[part_idx].append(i)
        cnt -= 1
        if cnt == 0:
            rel_dict[r].pop(0)
        else:
            rel_dict[r][0][1] = cnt

    for i, part in enumerate(parts):
        parts[i] = np.array(part, dtype=np.int64)
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]

    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)

    return parts, rel_parts, num_cross_part > 0

def RandomPartition(edges, n):
    """This partitions a list of edges randomly across n partitions

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list to partition
    n : int
        number of partitions

    Returns
    -------
    List of np.array
        Edges of each partition
    """
    heads, rels, tails = edges
    print('random partition {} edges into {} parts'.format(len(heads), n))
    idx = np.random.permutation(len(heads))
    heads[:] = heads[idx]
    rels[:] = rels[idx]
    tails[:] = tails[idx]

    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append(idx[start:end])
        print('part {} has {} edges'.format(i, len(parts[-1])))
    return parts

def ConstructGraph(edges, n_entities, args):
    """Construct Graph for training

    Parameters
    ----------
    edges : (heads, rels, tails) triple
        Edge list
    n_entities : int
        number of entities
    args :
        Global configs.
    """
    pickle_name = 'graph_train.pickle'
    if args.pickle_graph and os.path.exists(os.path.join(args.data_path, args.dataset, pickle_name)):
        with open(os.path.join(args.data_path, args.dataset, pickle_name), 'rb') as graph_file:
            g = pickle.load(graph_file)
            print('Load pickled graph.')
    else:
        src, etype_id, dst = edges
        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
        g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        if args.pickle_graph:
            with open(os.path.join(args.data_path, args.dataset, pickle_name), 'wb') as graph_file:
                pickle.dump(g, graph_file)
    return g

class TrainDataset(object):
    """Dataset for training

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    ranks:
        Number of partitions.
    """
    def __init__(self, dataset, args, ranks=64):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)

        if ranks > 1 and args.soft_rel_part:
            self.edge_parts, self.rel_parts, self.cross_part, self.cross_rels = \
            SoftRelationPartition(triples, ranks)
        elif ranks > 1 and args.rel_part:
            self.edge_parts, self.rel_parts, self.cross_part = \
                BalancedRelationPartition(triples, ranks)
        elif ranks > 1:
            self.edge_parts = RandomPartition(triples, ranks)
            self.cross_part = True
        else:
            self.edge_parts = [np.arange(num_train)]
            self.rel_parts = [np.arange(dataset.n_relations)]
            self.cross_part = False

        self.g = ConstructGraph(triples, dataset.n_entities, args)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=32,
                       shuffle=True, exclude_positive=False, rank=0):
        """Create sampler for training

        Parameters
        ----------
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        shuffle : bool
            If True, shuffle the seed edges.
            If False, do not shuffle the seed edges.
            Default: False
        exclude_positive : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
            Default: False
        rank : int
            Which partition to sample.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        assert batch_size % neg_sample_size == 0, 'batch_size should be divisible by B'
        return EdgeSampler(self.g,
                           seed_edges=F.tensor(self.edge_parts[rank]),
                           batch_size=batch_size,
                           neg_sample_size=int(neg_sample_size/neg_chunk_size),
                           chunk_size=neg_chunk_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)


class ChunkNegEdgeSubgraph(dgl.DGLGraph):
    """Wrapper for negative graph

        Parameters
        ----------
        neg_g : DGLGraph
            Graph holding negative edges.
        num_chunks : int
            Number of chunks in sampled graph.
        chunk_size : int
            Info of chunk_size.
        neg_sample_size : int
            Info of neg_sample_size.
        neg_head : bool
            If True, negative_mode is 'head'
            If False, negative_mode is 'tail'
    """
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(graph_data=subg.sgi.graph,
                                                   readonly=True,
                                                   parent=subg._parent)
        self.ndata[NID] = subg.sgi.induced_nodes.tousertensor()
        self.edata[EID] = subg.sgi.induced_edges.tousertensor()
        self.subg = subg
        self.num_chunks = num_chunks
        self.chunk_size = chunk_size
        self.neg_sample_size = neg_sample_size
        self.neg_head = neg_head

    @property
    def head_nid(self):
        return self.subg.head_nid

    @property
    def tail_nid(self):
        return self.subg.tail_nid


def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked,
                        neg_head, num_nodes):
    """KG models need to know the number of chunks, the chunk size and negative sample size
    of a negative subgraph to perform the computation more efficiently.
    This function tries to infer all of these information of the negative subgraph
    and create a wrapper class that contains all of the information.

    Parameters
    ----------
    pos_g : DGLGraph
        Graph holding positive edges.
    neg_g : DGLGraph
        Graph holding negative edges.
    chunk_size : int
        Chunk size of negative subgrap.
    neg_sample_size : int
        Negative sample size of negative subgrap.
    is_chunked : bool
        If True, the sampled batch is chunked.
    neg_head : bool
        If True, negative_mode is 'head'
        If False, negative_mode is 'tail'
    num_nodes: int
        Total number of nodes in the whole graph.

    Returns
    -------
    ChunkNegEdgeSubgraph
        Negative graph wrapper
    """
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
            or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        # This is probably for evaluation.
        if pos_g.number_of_edges() < chunk_size \
                and neg_g.number_of_edges() % neg_sample_size == 0:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        # This is probably the last batch in the training. Let's ignore it.
        elif pos_g.number_of_edges() % chunk_size > 0:
            return None
        else:
            num_chunks = int(pos_g.number_of_edges() / chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)

class EvalSampler(object):
    """Sampler for validation and testing

    Parameters
    ----------
    g : DGLGraph
        Graph containing KG graph
    edges : tensor
        Seed edges
    batch_size : int
        Batch size of each mini batch.
    neg_sample_size : int
        How many negative edges sampled for each node.
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    mode : str
        Sampling mode.
    number_workers: int
        Number of workers used in parallel for this sampler
    filter_false_neg : bool
        If True, exlucde true positive edges in sampled negative edges
        If False, return all sampled negative edges even there are positive edges
        Default: True
    """
    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers=32,
                 filter_false_neg=True):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   chunk_size=neg_chunk_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['tid'],
                                   return_false_neg=filter_false_neg)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g
        self.filter_false_neg = filter_false_neg
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size

    def __iter__(self):
        return self

    def __next__(self):
        """Get next batch

        Returns
        -------
        DGLGraph
            Sampled positive graph
        ChunkNegEdgeSubgraph
            Negative graph wrapper
        """
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g, 
                                        self.neg_chunk_size, 
                                        self.neg_sample_size, 
                                        'chunk' in self.mode, 
                                        self.neg_head, 
                                        self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        """Reset the sampler
        """
        self.sampler_iter = iter(self.sampler)
        return self

class EvalDataset(object):
    """Dataset for validation or testing

    Parameters
    ----------
    dataset : KGDataset
        Original dataset.
    args :
        Global configs.
    """
    def __init__(self, dataset, args):
        pickle_name = 'graph_all.pickle'
        if args.pickle_graph and os.path.exists(os.path.join(args.data_path, args.dataset, pickle_name)):
            with open(os.path.join(args.data_path, args.dataset, pickle_name), 'rb') as graph_file:
                g = pickle.load(graph_file)
                print('Load pickled graph.')
        else:
            src = np.concatenate((dataset.train[0], dataset.valid[0], dataset.test[0]))
            etype_id = np.concatenate((dataset.train[1], dataset.valid[1], dataset.test[1]))
            dst = np.concatenate((dataset.train[2], dataset.valid[2], dataset.test[2]))
            coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)),
                                       shape=[dataset.n_entities, dataset.n_entities])
            g = dgl.DGLGraph(coo, readonly=True, multigraph=True, sort_csr=True)
            g.edata['tid'] = F.tensor(etype_id, F.int64)
            if args.pickle_graph:
                with open(os.path.join(args.data_path, args.dataset, pickle_name), 'wb') as graph_file:
                    pickle.dump(g, graph_file)
        self.g = g
        self.num_train = len(dataset.train[0])
        self.num_valid = len(dataset.valid[0])
        self.num_test = len(dataset.test[0])

        if args.eval_percent < 1:
            self.valid = np.random.randint(0, self.num_valid,
                    size=(int(self.num_valid * args.eval_percent),)) + self.num_train
        else:
            self.valid = np.arange(self.num_train, self.num_train + self.num_valid)
        print('|valid|:', len(self.valid))

        if args.eval_percent < 1:
            self.test = np.random.randint(0, self.num_test,
                    size=(int(self.num_test * args.eval_percent,)))
            self.test += self.num_train + self.num_valid
        else:
            self.test = np.arange(self.num_train + self.num_valid, self.g.number_of_edges())
        print('|test|:', len(self.test))

    def get_edges(self, eval_type):
        """ Get all edges in this dataset

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing

        Returns
        -------
        np.array
            Edges
        """
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=32, rank=0, ranks=1):
        """Create sampler for validation or testing

        Parameters
        ----------
        eval_type : str
            Sampling type, 'valid' for validation and 'test' for testing
        batch_size : int
            Batch size of each mini batch.
        neg_sample_size : int
            How many negative edges sampled for each node.
        neg_chunk_size : int
            How many edges in one chunk. We split one batch into chunks.
        filter_false_neg : bool
            If True, exlucde true positive edges in sampled negative edges
            If False, return all sampled negative edges even there are positive edges
        mode : str
            Sampling mode.
        number_workers: int
            Number of workers used in parallel for this sampler
        rank : int
            Which partition to sample.
        ranks : int
            Total number of partitions.

        Returns
        -------
        dgl.contrib.sampling.EdgeSampler
            Edge sampler
        """
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

class NewBidirectionalOneShotIterator:
    """Grouped samper iterator

    Parameters
    ----------
    dataloader_head : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in head mode
    dataloader_tail : dgl.contrib.sampling.EdgeSampler
        EdgeSampler in tail mode
    neg_chunk_size : int
        How many edges in one chunk. We split one batch into chunks.
    neg_sample_size : int
        How many negative edges sampled for each node.
    is_chunked : bool
        If True, the sampled batch is chunked.
    num_nodes : int
        Total number of nodes in the whole graph.
    """
    def __init__(self, dataloader_head, dataloader_tail, neg_chunk_size, neg_sample_size,
                 is_chunked, num_nodes):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    True, num_nodes)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, neg_chunk_size,
                                                    neg_sample_size, is_chunked,
                                                    False, num_nodes)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            pos_g, neg_g = next(self.iterator_head)
        else:
            pos_g, neg_g = next(self.iterator_tail)
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked,
                          neg_head, num_nodes):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size,
                                            is_chunked, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.ndata['id'] = pos_g.parent_nid
                neg_g.ndata['id'] = neg_g.parent_nid
                pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
                yield pos_g, neg_g
