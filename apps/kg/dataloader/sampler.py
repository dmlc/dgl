import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import sys
import pickle
import time
import multiprocessing as mp

# Given a list of relations and the number of edges that belong to the relation,
# we greedily assign relations and edges to a set of partitions to ensure that
# each partition has the same number of edges and roughly the same number of
# relations.
def greedy_assign(uniq, cnts, edge_cnts, rel_cnts, rel_dict, rel_parts):
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        idx = np.argmin(edge_cnts)
        rel_dict[r] = idx
        rel_parts[idx].append(r)
        edge_cnts[idx] += cnt
        rel_cnts[idx] += 1

def BucketRelationPartition(edges, n):
    heads, rels, tails = edges
    uniq, cnts = np.unique(rels, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    num_edges = len(rels)

    part_size = int(num_edges / n)
    # If a relation has edges more than half of a partition size, we consider this relation
    # a giant relation.
    idx = cnts > (part_size / 10)
    giant_rels = uniq[idx]

    # Get the remaining relations.
    idx = np.logical_not(idx)
    cnts = cnts[idx]
    uniq = uniq[idx]

    # split the relations into buckets.
    buckets = []
    size = 16
    while len(cnts) > 0:
        cnts1 = cnts[:size]
        uniq1 = uniq[:size]
        cnts = cnts[size:]
        uniq = uniq[size:]
        buckets.append((cnts1, uniq1))
        size *= 2

    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    rel_parts = []
    for _ in range(n):
        rel_parts.append([])

    for cnts1, uniq1 in buckets:
        # random permute relations.
        rand_idx = np.random.permutation(len(cnts1))
        cnts1 = cnts1[rand_idx]
        uniq1 = uniq1[rand_idx]
        greedy_assign(uniq1, cnts1, edge_cnts, rel_cnts, rel_dict, rel_parts)

    parts = []
    for i in range(n):
        parts.append([])
    # We partition giant relations and evenly split them into all partitions.
    # Here we assume we don't have many giant relations.
    selected_idx = np.zeros(len(rels))
    print('there are {} giant relations'.format(len(giant_rels)))
    for giant_rel in giant_rels:
        idx = (rels == giant_rel)
        selected_idx += idx
        giant_eids = np.nonzero(idx)[0]
        giant_eids = np.random.permutation(giant_eids)
        step_size = int(len(giant_eids)/n + 1)
        giant_parts = np.split(giant_eids, np.arange(step_size, len(giant_eids), step_size))
        for i, giant_part in enumerate(giant_parts):
            parts[i].append(giant_part)

    # Here we store unpopular relations into partitions. Each relation appears
    # in one of the partition.
    small_parts = []
    for i in range(n):
        small_parts.append([])
        rel_parts[i] = np.array(rel_parts[i])
    # let's store the edge index to each partition first.
    for eid in np.nonzero(selected_idx == 0)[0]:
        r = rels[eid]
        part_idx = rel_dict[r]
        small_parts[part_idx].append(eid)
    for i, part in enumerate(small_parts):
        small_parts[i] = np.array(part, dtype=np.int64)

    # Put all edges together in each partition.
    for i, part in enumerate(parts):
        small_part = small_parts[i]
        part.append(small_part)
        part = np.concatenate(part)
        parts[i] = part
        print('part {} has {} edges and {} relations'.format(i, len(part), len(rel_parts[i]) + len(giant_rels)))

    # shuffle edges
    shuffle_idx = np.concatenate(parts)
    heads[:] = heads[shuffle_idx]
    rels[:] = rels[shuffle_idx]
    tails[:] = tails[shuffle_idx]
    off = 0
    for i, part in enumerate(parts):
        parts[i] = np.arange(off, off + len(part))
        off += len(part)

    return parts


# This partitions a list of edges based on relations and make sure
# each relation only fall into one partition.
def StrictRelationPartition(edges, n):
    heads, rels, tails = edges
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
    greedy_assign(uniq, cnts, edge_cnts, rel_cnts, rel_dict, rel_parts)
    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))

    parts = []
    for i in range(n):
        parts.append([])
        rel_parts[i] = np.array(rel_parts[i])
    # let's store the edge index to each partition first.
    for i, r in enumerate(rels):
        part_idx = rel_dict[r]
        parts[part_idx].append(i)
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

    return parts, rel_parts

# This partitions a list of edges based on relations to make sure
# each partition has roughly the same number of edges and relations.
def BalancedRelationPartition(edges, n):
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
    num_div = 0
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
                num_div += 1
                cnt -= cur_cnt
        rel_dict[r] = r_parts

    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    print('{}/{} duplicated relation across partitions'.format(num_div, len(cnts)))

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
        print(parts[i].shape)
        
    # TODO(zhengda) we should also reshuffle here to speed up
    return parts, rel_parts

def RandomPartition(edges, n):
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
        parts.append(np.arange(start, end))
        print('part {} has {} edges'.format(i, len(parts[-1])))
    return parts

def ConstructGraph(edges, n_entities, args):
    pickle_name = 'graph_train.pickle'
    if args.pickle_graph and os.path.exists(os.path.join(args.data_path, args.dataset, pickle_name)):
        with open(os.path.join(args.data_path, args.dataset, pickle_name), 'rb') as graph_file:
            g = pickle.load(graph_file)
            print('Load pickled graph.')
    else:
        src, etype_id, dst = edges
        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
        g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
        g.edata['tid'] = F.tensor(etype_id, F.int64)
        if args.pickle_graph:
            with open(os.path.join(args.data_path, args.dataset, pickle_name), 'wb') as graph_file:
                pickle.dump(g, graph_file)
    return g

class TrainDataset(object):
    def __init__(self, dataset, args, weighting=False, ranks=64):
        triples = dataset.train
        num_train = len(triples[0])
        print('|Train|:', num_train)

        if ranks > 1 and args.strict_rel_part:
            self.edge_parts, self.rel_parts = StrictRelationPartition(triples, ranks)
        elif ranks > 1 and args.rel_part:
            self.edge_parts = lambda: BucketRelationPartition(triples, ranks)
        elif ranks > 1:
            self.edge_parts = lambda: RandomPartition(triples, ranks)
        else:
            self.edge_parts = [np.arange(num_train)]
        self.g = ConstructGraph(triples, dataset.n_entities, args)
        if weighting:
            # TODO: weight to be added
            count = self.count_freq(triples)
            subsampling_weight = np.vectorize(
                lambda h, r, t: np.sqrt(1 / (count[(h, r)] + count[(t, -r - 1)]))
            )
            weight = subsampling_weight(src, etype_id, dst)
            self.g.edata['weight'] = F.zerocopy_from_numpy(weight)

    def count_freq(self, triples, start=4):
        count = {}
        for head, rel, tail in triples:
            if (head, rel) not in count:
                count[(head, rel)] = start
            else:
                count[(head, rel)] += 1

            if (tail, -rel - 1) not in count:
                count[(tail, -rel - 1)] = start
            else:
                count[(tail, -rel - 1)] += 1
        return count

    def get_sampler_server(self, num_proc, num_epochs):
        return SamplerServer(self.g, num_proc, num_epochs, self.edge_parts)

    def create_sampler(self, batch_size, neg_sample_size=2, neg_chunk_size=None, mode='head', num_workers=5,
                       shuffle=True, exclude_positive=False, rank=0):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
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


class SamplerServer:
    def __init__(self, g, num_proc, num_epochs, partition_edges):
        self.g = g
        self.queues = []
        for _ in range(num_proc):
            self.queues.append([])

        for i in range(num_epochs):
            edge_parts = partition_edges()
            for i, part in enumerate(edge_parts):
                self.queues[i].append(part)

    def get_creator(self, batch_size, rank, neg_sample_size=2, neg_chunk_size=None, num_workers=5):
        return SamplerCreator(self.queues[rank], self.g, batch_size, neg_sample_size, neg_chunk_size, num_workers)

class SamplerCreator:
    def __init__(self, queue, g, batch_size, neg_sample_size=2, neg_chunk_size=None, num_workers=5):
        self.queue = queue
        self.g = g
        self.batch_size = batch_size
        self.neg_sample_size = neg_sample_size
        self.neg_chunk_size = neg_chunk_size
        self.num_workers = num_workers
        self.epoch = 0

    def create(self):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        edge_part = self.queue[0]
        self.epoch += 1
        sampler_head = EdgeSampler(self.g,
                              seed_edges=edge_part,
                              batch_size=self.batch_size,
                              neg_sample_size=int(self.neg_sample_size/self.neg_chunk_size),
                              chunk_size=self.neg_chunk_size,
                              negative_mode='head',
                              num_workers=self.num_workers,
                              shuffle=True,
                              exclude_positive=False,
                              return_false_neg=False)
        sampler_tail = EdgeSampler(self.g,
                              seed_edges=edge_part,
                              batch_size=self.batch_size,
                              neg_sample_size=int(self.neg_sample_size/self.neg_chunk_size),
                              chunk_size=self.neg_chunk_size,
                              negative_mode='tail',
                              num_workers=self.num_workers,
                              shuffle=True,
                              exclude_positive=False,
                              return_false_neg=False)
        return sampler_head, sampler_tail


class ChunkNegEdgeSubgraph(dgl.subgraph.DGLSubGraph):
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(ChunkNegEdgeSubgraph, self).__init__(subg._parent, subg.sgi)
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


# KG models need to know the number of chunks, the chunk size and negative sample size
# of a negative subgraph to perform the computation more efficiently.
# This function tries to infer all of these information of the negative subgraph
# and create a wrapper class that contains all of the information.
def create_neg_subgraph(pos_g, neg_g, chunk_size, neg_sample_size, is_chunked, neg_head, num_nodes):
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
       or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_chunked:
        if pos_g.number_of_edges() < chunk_size:
            return None
        else:
            # This is probably the last batch. Let's ignore it.
            if pos_g.number_of_edges() % chunk_size > 0:
                return None
            num_chunks = int(pos_g.number_of_edges()/ chunk_size)
        assert num_chunks * chunk_size == pos_g.number_of_edges()
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return ChunkNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                                neg_sample_size, neg_head)

class EvalSampler(object):
    def __init__(self, g, edges, batch_size, neg_sample_size, neg_chunk_size, mode, num_workers,
                 filter_false_neg):
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
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            if self.filter_false_neg:
                neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g, self.neg_chunk_size, self.neg_sample_size, 'chunk' in self.mode,
                                        self.neg_head, self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.ndata['id'] = pos_g.parent_nid
        neg_g.ndata['id'] = neg_g.parent_nid
        pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
        if self.filter_false_neg:
            neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        self.sampler_iter = iter(self.sampler)
        return self

class EvalDataset(object):
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
            g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
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

        self.num_valid = len(self.valid)
        self.num_test = len(self.test)

    def get_edges(self, eval_type):
        if eval_type == 'valid':
            return self.valid
        elif eval_type == 'test':
            return self.test
        else:
            raise Exception('get invalid type: ' + eval_type)

    def check(self, eval_type):
        edges = self.get_edges(eval_type)
        subg = self.g.edge_subgraph(edges)
        if eval_type == 'valid':
            data = self.valid
        elif eval_type == 'test':
            data = self.test

        subg.copy_from_parent()
        src, dst, eid = subg.all_edges('all', order='eid')
        src_id = subg.ndata['id'][src]
        dst_id = subg.ndata['id'][dst]
        etype = subg.edata['id'][eid]

        orig_src = np.array([t[0] for t in data])
        orig_etype = np.array([t[1] for t in data])
        orig_dst = np.array([t[2] for t in data])
        np.testing.assert_equal(F.asnumpy(src_id), orig_src)
        np.testing.assert_equal(F.asnumpy(dst_id), orig_dst)
        np.testing.assert_equal(F.asnumpy(etype), orig_etype)

    def create_sampler(self, eval_type, batch_size, neg_sample_size, neg_chunk_size,
                       filter_false_neg, mode='head', num_workers=5, rank=0, ranks=1):
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, neg_chunk_size,
                           mode, num_workers, filter_false_neg)

class NewBidirectionalOneShotIterator:
    def __init__(self, sample_creator, neg_chunk_size, neg_sample_size, is_chunked, num_nodes, rank):
        self.step = 0
        self.neg_chunk_size = neg_chunk_size
        self.neg_sample_size = neg_sample_size
        self.is_chunked = is_chunked
        self.num_nodes = num_nodes
        self.sample_creator = sample_creator
        self.rank = rank
        self.epoch = 0
        self.reset()

    def reset(self):
        self.sampler_head, self.sampler_tail = self.sample_creator.create()
        self.iterator_head = self.one_shot_iterator(self.sampler_head, self.neg_chunk_size, self.neg_sample_size,
                                                    self.is_chunked, True, self.num_nodes)
        self.iterator_tail = self.one_shot_iterator(self.sampler_tail, self.neg_chunk_size, self.neg_sample_size,
                                                    self.is_chunked, False, self.num_nodes)
        self.epoch += 1

    def __next__(self):
        self.step += 1
        start = time.time()
        pos_g = None
        neg_g = None
        while pos_g is None:
            try:
                if self.step % 2 == 0:
                    pos_g, neg_g = next(self.iterator_head)
                else:
                    pos_g, neg_g = next(self.iterator_tail)
            except:
                self.reset()
        return pos_g, neg_g

    @staticmethod
    def one_shot_iterator(dataloader, neg_chunk_size, neg_sample_size, is_chunked, neg_head, num_nodes):
        for pos_g, neg_g in dataloader:
            neg_g = create_neg_subgraph(pos_g, neg_g, neg_chunk_size, neg_sample_size, is_chunked,
                                        neg_head, num_nodes)
            if neg_g is None:
                return None, None

            pos_g.ndata['id'] = pos_g.parent_nid
            neg_g.ndata['id'] = neg_g.parent_nid
            pos_g.edata['id'] = pos_g._parent.edata['tid'][pos_g.parent_eid]
            yield pos_g, neg_g

