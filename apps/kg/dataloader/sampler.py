import math
import numpy as np
import scipy as sp
import dgl.backend as F
import dgl
import os
import pickle
import time

# This partitions a list of edges based on relations to make sure
# each partition has roughly the same number of edges and relations.
def RelationPartition(edges, n):
    print('relation partition {} edges into {} parts'.format(len(edges), n))
    rel = np.array([r for h, r, t in edges])
    uniq, cnts = np.unique(rel, return_counts=True)
    idx = np.flip(np.argsort(cnts))
    cnts = cnts[idx]
    uniq = uniq[idx]
    assert cnts[0] > cnts[-1]
    edge_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_cnts = np.zeros(shape=(n,), dtype=np.int64)
    rel_dict = {}
    for i in range(len(cnts)):
        cnt = cnts[i]
        r = uniq[i]
        idx = np.argmin(edge_cnts)
        rel_dict[r] = idx
        edge_cnts[idx] += cnt
        rel_cnts[idx] += 1
    for i, edge_cnt in enumerate(edge_cnts):
        print('part {} has {} edges and {} relations'.format(i, edge_cnt, rel_cnts[i]))
    parts = []
    for _ in range(n):
        parts.append([])
    for h, r, t in edges:
        idx = rel_dict[r]
        parts[idx].append((h, r, t))
    return parts

def RandomPartition(edges, n):
    print('random partition {} edges into {} parts'.format(len(edges), n))
    idx = np.random.permutation(len(edges))
    part_size = int(math.ceil(len(idx) / n))
    parts = []
    for i in range(n):
        start = part_size * i
        end = min(part_size * (i + 1), len(idx))
        parts.append([edges[i] for i in idx[start:end]])
    return parts

def ConstructGraph(edges, n_entities, i, args):
    pickle_name = 'graph_train_{}.pickle'.format(i)
    if args.pickle_graph and os.path.exists(os.path.join(args.data_path, args.dataset, pickle_name)):
        with open(os.path.join(args.data_path, args.dataset, pickle_name), 'rb') as graph_file:
            g = pickle.load(graph_file)
            print('Load pickled graph.')
    else:
        src = [t[0] for t in edges]
        etype_id = [t[1] for t in edges]
        dst = [t[2] for t in edges]
        coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[n_entities, n_entities])
        g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
        g.ndata['id'] = F.arange(0, g.number_of_nodes())
        g.edata['id'] = F.tensor(etype_id, F.int64)
        if args.pickle_graph:
            with open(os.path.join(args.data_path, args.dataset, pickle_name), 'wb') as graph_file:
                pickle.dump(g, graph_file)
    return g

class TrainDataset(object):
    def __init__(self, dataset, args, weighting=False, ranks=64):
        triples = dataset.train
        print('|Train|:', len(triples))
        if ranks > 1 and args.rel_part:
            triples_list = RelationPartition(triples, ranks)
        elif ranks > 1:
            triples_list = RandomPartition(triples, ranks)
        else:
            triples_list = [triples]
        self.graphs = []
        for i, triples in enumerate(triples_list):
            g = ConstructGraph(triples, dataset.n_entities, i, args)
            if weighting:
                # TODO: weight to be added
                count = self.count_freq(triples)
                subsampling_weight = np.vectorize(
                    lambda h, r, t: np.sqrt(1 / (count[(h, r)] + count[(t, -r - 1)]))
                )
                weight = subsampling_weight(src, etype_id, dst)
                g.edata['weight'] = F.zerocopy_from_numpy(weight)
                # to be added
            self.graphs.append(g)

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

    def create_sampler(self, batch_size, neg_sample_size=2, mode='head', num_workers=5,
                       shuffle=True, exclude_positive=False, rank=0):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        return EdgeSampler(self.graphs[rank],
                           batch_size=batch_size,
                           neg_sample_size=neg_sample_size,
                           negative_mode=mode,
                           num_workers=num_workers,
                           shuffle=shuffle,
                           exclude_positive=exclude_positive,
                           return_false_neg=False)

class PBGNegEdgeSubgraph(dgl.subgraph.DGLSubGraph):
    def __init__(self, subg, num_chunks, chunk_size,
                 neg_sample_size, neg_head):
        super(PBGNegEdgeSubgraph, self).__init__(subg._parent, subg.sgi)
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


def create_neg_subgraph(pos_g, neg_g, is_pbg, neg_head, num_nodes):
    assert neg_g.number_of_edges() % pos_g.number_of_edges() == 0
    neg_sample_size = int(neg_g.number_of_edges() / pos_g.number_of_edges())
    # We use all nodes to create negative edges. Regardless of the sampling algorithm,
    # we can always view the subgraph with one chunk.
    if (neg_head and len(neg_g.head_nid) == num_nodes) \
       or (not neg_head and len(neg_g.tail_nid) == num_nodes):
        num_chunks = 1
        chunk_size = pos_g.number_of_edges()
    elif is_pbg:
        if pos_g.number_of_edges() < neg_sample_size:
            num_chunks = 1
            chunk_size = pos_g.number_of_edges()
        else:
            # This is probably the last batch. Let's ignore it.
            if pos_g.number_of_edges() % neg_sample_size > 0:
                return None
            num_chunks = int(pos_g.number_of_edges()/ neg_sample_size)
            chunk_size = neg_sample_size
    else:
        num_chunks = pos_g.number_of_edges()
        chunk_size = 1
    return PBGNegEdgeSubgraph(neg_g, num_chunks, chunk_size,
                              neg_sample_size, neg_head)

class EvalSampler(object):
    def __init__(self, g, edges, batch_size, neg_sample_size, mode, num_workers):
        EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
        self.sampler = EdgeSampler(g,
                                   batch_size=batch_size,
                                   seed_edges=edges,
                                   neg_sample_size=neg_sample_size,
                                   negative_mode=mode,
                                   num_workers=num_workers,
                                   shuffle=False,
                                   exclude_positive=False,
                                   relations=g.edata['id'],
                                   return_false_neg=True)
        self.sampler_iter = iter(self.sampler)
        self.mode = mode
        self.neg_head = 'head' in mode
        self.g = g

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            pos_g, neg_g = next(self.sampler_iter)
            neg_positive = neg_g.edata['false_neg']
            neg_g = create_neg_subgraph(pos_g, neg_g, 'PBG' in self.mode,
                                        self.neg_head, self.g.number_of_nodes())
            if neg_g is not None:
                break

        pos_g.copy_from_parent()
        neg_g.copy_from_parent()
        neg_g.edata['bias'] = F.astype(-neg_positive, F.float32)
        return pos_g, neg_g

    def reset(self):
        self.sampler_iter = iter(self.sampler)
        return self

class EvalDataset(object):
    def __init__(self, dataset, args):
        triples = dataset.train + dataset.valid + dataset.test
        pickle_name = 'graph_all.pickle'
        if args.pickle_graph and os.path.exists(os.path.join(args.data_path, args.dataset, pickle_name)):
            with open(os.path.join(args.data_path, args.dataset, pickle_name), 'rb') as graph_file:
                g = pickle.load(graph_file)
                print('Load pickled graph.')
        else:
            src = [t[0] for t in triples]
            etype_id = [t[1] for t in triples]
            dst = [t[2] for t in triples]
            coo = sp.sparse.coo_matrix((np.ones(len(src)), (src, dst)), shape=[dataset.n_entities, dataset.n_entities])
            g = dgl.DGLGraph(coo, readonly=True, sort_csr=True)
            g.ndata['id'] = F.arange(0, g.number_of_nodes())
            g.edata['id'] = F.tensor(etype_id, F.int64)
            if args.pickle_graph:
                with open(os.path.join(args.data_path, args.dataset, pickle_name), 'wb') as graph_file:
                    pickle.dump(g, graph_file)
        self.g = g

        self.num_train = len(dataset.train)
        self.num_valid = len(dataset.valid)
        self.num_test = len(dataset.test)

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

    def create_sampler(self, eval_type, batch_size, neg_sample_size, mode='head',
                       num_workers=5, rank=0, ranks=1):
        edges = self.get_edges(eval_type)
        beg = edges.shape[0] * rank // ranks
        end = min(edges.shape[0] * (rank + 1) // ranks, edges.shape[0])
        edges = edges[beg: end]
        print("eval on {} edges".format(len(edges)))
        return EvalSampler(self.g, edges, batch_size, neg_sample_size, mode, num_workers)

class NewBidirectionalOneShotIterator:
    def __init__(self, dataloader_head, dataloader_tail, is_pbg, num_nodes):
        self.sampler_head = dataloader_head
        self.sampler_tail = dataloader_tail
        self.iterator_head = self.one_shot_iterator(dataloader_head, is_pbg,
                                                    True, num_nodes)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail, is_pbg,
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
    def one_shot_iterator(dataloader, is_pbg, neg_head, num_nodes):
        while True:
            for pos_g, neg_g in dataloader:
                neg_g = create_neg_subgraph(pos_g, neg_g, is_pbg, neg_head, num_nodes)
                if neg_g is None:
                    continue

                pos_g.copy_from_parent()
                neg_g.copy_from_parent()
                yield pos_g, neg_g
