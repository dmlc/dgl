import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import ndarray as nd

class TransEScore(nn.Block):
    def __init__(self, gamma):
        super(TransEScore, self).__init__()
        self.gamma = gamma

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - nd.norm(score, ord=1, axis=-1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks,chunk_size, 1, hidden_dim)
                return gamma - nd.norm(heads - tails, ord=1, axis=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads + relations
                heads = heads.reshape(num_chunks, chunk_size, 1, hidden_dim)
                tails = tails.reshape(num_chunks, 1, neg_sample_size, hidden_dim)
                return gamma - nd.norm(heads - tails, ord=1, axis=-1)
            return fn

class DistMultScore(nn.Block):
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': nd.sum(score, axis=-1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = nd.transpose(heads, axes=(0, 2, 1))
                tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = nd.transpose(tails, axes=(0, 2, 1))
                tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, tails)
            return fn

class RESCALScore(nn.Block):
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].expand_dims(2)
        rel = edges.data['emb']
        rel = rel.reshape(-1, self.relation_dim, self.entity_dim)
        score = head * mx.nd.batch_dot(rel, tail).squeeze()
        # TODO: check if use self.gamma
        return {'score': mx.nd.sum(score, -1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head):
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = mx.nd.transpose(heads, axes=(0,2,1))
                tails = tails.expand_dims(2)
                relations = relations.reshape(-1, self.relation_dim, self.entity_dim)
                tmp = mx.nd.batch_dot(relations, tails).squeeze()
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = mx.nd.transpose(tails, axes=(0,2,1))
                heads = heads.expand_dims(2)
                relations = relations.reshape(-1, self.relation_dim, self.entity_dim)
                tmp = mx.nd.batch_dot(relations, heads).squeeze()
                tmp = tmp.reshape(num_chunks, chunk_size, hidden_dim)
                return nd.linalg_gemm2(tmp, tails)
            return fn