import os
import numpy as np

class BasePBGKEModel(object):
    def __init__(self, score_func, batch_size, neg_sample_size, hidden_dim, n_entities):
        self.score_func = score_func
        head_neg_score, tail_neg_score = self.create_neg(batch_size,
                                                         neg_sample_size,
                                                         hidden_dim,
                                                         n_entities)
        self.head_neg_score = head_neg_score
        self.tail_neg_score = tail_neg_score
        self.neg_sample_size = neg_sample_size

    def create_neg(self, batch_size, neg_sample_size, hidden_dim, n_entities):
        if neg_sample_size > 0:
            chunk_size = min(neg_sample_size, batch_size)
            num_chunks = int(batch_size / chunk_size)
        else:
            chunk_size = batch_size
            num_chunks = 1
            # Here we are using all nodes to create negative edges.
            neg_sample_size = n_entities

        head_neg_score = self.score_func.create_neg(True, num_chunks,
                                                    chunk_size,
                                                    neg_sample_size,
                                                    hidden_dim)
        tail_neg_score = self.score_func.create_neg(False, num_chunks,
                                                    chunk_size,
                                                    neg_sample_size,
                                                    hidden_dim)
        return head_neg_score, tail_neg_score

    def predict_score(self, g):
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, entity_emb, neg_head, to_device=None, gpu_id=-1, trace=False):
        neg_sample_size = self.head_neg_score.neg_sample_size
        num_chunks = self.head_neg_score.num_chunks
        chunk_size = self.head_neg_score.chunk_size

        if pos_g.number_of_edges() != num_chunks * chunk_size:
            return self.predict_score(neg_g)

        assert neg_g.number_of_edges() == pos_g.number_of_edges() * neg_sample_size
        if neg_head:
            neg_head_ids, _ = neg_g.all_edges(order='eid')
            neg_head_ids = neg_g.ndata['id'][neg_head_ids]
            # TODO this hack isn't necessary when bipartite graph is supported.
            neg_head_ids = neg_head_ids.reshape(num_chunks, chunk_size, neg_sample_size)
            neg_head_ids = neg_head_ids[:,0,:].reshape(num_chunks * neg_sample_size)
            neg_head = entity_emb(neg_head_ids, gpu_id, trace)

            _, tail_ids = pos_g.all_edges(order='eid')
            if to_device is not None:
                tail_ids = to_device(tail_ids, gpu_id)
            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']
            neg_score = self.head_neg_score(neg_head, rel, tail)
        else:
            _, neg_tail_ids = neg_g.all_edges(order='eid')
            neg_tail_ids = neg_g.ndata['id'][neg_tail_ids]
            # TODO this hack isn't necessary when bipartite graph is supported.
            neg_tail_ids = neg_tail_ids.reshape(num_chunks, chunk_size, neg_sample_size)
            neg_tail_ids = neg_tail_ids[:,0,:].reshape(num_chunks * neg_sample_size)
            neg_tail = entity_emb(neg_tail_ids, gpu_id, trace)

            head_ids, _ = pos_g.all_edges(order='eid')
            if to_device is not None:
                head_ids = to_device(head_ids, gpu_id)
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']
            neg_score = self.tail_neg_score(head, rel, neg_tail)

        return neg_score

class BaseKEModel(object):
    def __init__(self, score_func, batch_size, neg_sample_size, hidden_dim):
        self.score_func = score_func
        head_neg_score, tail_neg_score = self.create_neg(batch_size,
                                                         neg_sample_size,
                                                         hidden_dim)
        self.head_neg_score = head_neg_score
        self.tail_neg_score = tail_neg_score
        self.neg_sample_size = neg_sample_size

    def create_neg(self, batch_size, neg_sample_size, hidden_dim):
        head_neg_score = self.score_func.create_neg(True, batch_size,
                                                    1, neg_sample_size,
                                                    hidden_dim)
        tail_neg_score = self.score_func.create_neg(False, batch_size,
                                                    1, neg_sample_size,
                                                    hidden_dim)
        return head_neg_score, tail_neg_score

    def predict_score(self, g):
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g, entity_emb, neg_head, neg_sample_size, gpu_id=-1, trace=False):
        assert neg_sample_size == self.head_neg_score.neg_sample_size, 'Neg_sample_size not match'
        num_chunks = self.head_neg_score.num_chunks
        chunk_size = self.head_neg_score.chunk_size

        if pos_g.number_of_edges() != num_chunks * chunk_size:
            return self.predict_score(neg_g, trace)

        assert neg_g.number_of_edges() == pos_g.number_of_edges() * neg_sample_size
        if neg_head:
            neg_head_ids, _ = neg_g.all_edges(order='eid')
            neg_head_ids = neg_g.ndata['id'][neg_head_ids]
            neg_head = neg_g.ndata['emb'](neg_head_ids, gpu_id, trace)

            _, tail_ids = pos_g.all_edges(order='eid')
            tail = entity_emb(tail_ids, gpu_id, trace)
            rel = pos_g.edata['emb']
            neg_score = self.head_neg_score(neg_head, rel, tail)
        else:
            _, neg_tail_ids = neg_g.all_edges(order='eid')
            neg_tail_ids = neg_g.ndata['id'][neg_tail_ids]
            neg_tail = neg_g.ndata['emb'](neg_tail_ids, gpu_id, trace)

            head_ids, _ = pos_g.all_edges(order='eid')
            head = entity_emb(head_ids, gpu_id, trace)
            rel = pos_g.edata['emb']
            neg_score = self.tail_neg_score(head, rel, neg_tail)
        return neg_score
