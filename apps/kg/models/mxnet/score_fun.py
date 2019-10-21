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

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

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

class TransRScore(nn.Block):
    def __init__(self, gamma, projection_emb, relation_dim, entity_dim):
        super(TransRScore, self).__init__()
        self.gamma = gamma
        self.projection_emb = projection_emb
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.data['head_emb']
        tail = edges.data['tail_emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - nd.norm(score, ord=1, axis=-1)}

    def prepare(self, g, gpu_id, trace=False):
        head_ids, tail_ids = g.all_edges(order='eid')
        projection = self.projection_emb(g.edata['id'], gpu_id, trace)
        projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
        head_emb = g.ndata['emb'][head_ids.as_in_context(g.ndata['emb'].context)].expand_dims(axis=-2)
        tail_emb = g.ndata['emb'][tail_ids.as_in_context(g.ndata['emb'].context)].expand_dims(axis=-2)
        g.edata['head_emb'] = nd.batch_dot(head_emb, projection).squeeze()
        g.edata['tail_emb'] = nd.batch_dot(tail_emb, projection).squeeze()

    def create_neg_prepare(self, neg_head):
        if neg_head:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(-1, 1, self.entity_dim)
                tail = nd.batch_dot(tail, projection)
                tail = tail.reshape(num_chunks, -1, self.relation_dim)

                # neg node, each project to all relations
                projection = projection.reshape(num_chunks, -1, 1, self.entity_dim, self.relation_dim)
                head = head.reshape(num_chunks, 1, -1, 1, self.entity_dim)
                num_rels = projection.shape[1]
                num_nnodes = head.shape[2]
                projection = nd.broadcast_axis(projection, axis=2, size=num_nnodes)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                head = nd.broadcast_axis(head, axis=1, size=num_rels)
                head = head.reshape(-1, 1, self.entity_dim)
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                head = nd.batch_dot(head, projection).reshape(num_chunks, num_rels, num_nnodes, self.relation_dim)
                return head, tail
            return fn
        else:
            def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
                # pos node, project to its relation
                projection = self.projection_emb(rel_id, gpu_id, trace)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                head = head.reshape(-1, 1, self.entity_dim)
                head = nd.batch_dot(head, projection).squeeze()
                head = head.reshape(num_chunks, -1, self.relation_dim)

                projection = projection.reshape(num_chunks, -1, 1, self.entity_dim, self.relation_dim)
                tail = tail.reshape(num_chunks, 1, -1, 1, self.entity_dim)
                num_rels = projection.shape[1]
                num_nnodes = tail.shape[2]
                projection = nd.broadcast_axis(projection, axis=2, size=num_nnodes)
                projection = projection.reshape(-1, self.entity_dim, self.relation_dim)
                tail = nd.broadcast_axis(tail, axis=1, size=num_rels)
                tail = tail.reshape(-1, 1, self.entity_dim)
                # neg node, each project to all relations
                # (num_chunks, num_rel, num_neg_nodes, rel_dim)
                tail = nd.batch_dot(tail, projection).reshape(num_chunks, num_rels, num_nnodes, self.relation_dim)
                return head, tail
            return fn

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        self.projection_emb.init(1.0)

    def update(self):
        self.projection_emb.update()

    def save(self, path, name):
        self.projection_emb.save(path, name+'projection')

    def load(self, path, name):
        self.projection_emb.load(path, name+'projection')

    def create_neg(self, neg_head):
        gamma = self.gamma
        if neg_head:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                tails = tails - relations
                tails = tails.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - nd.norm(score, ord=1, axis=-1)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                relations = relations.reshape(num_chunks, -1, self.relation_dim)
                heads = heads - relations
                heads = heads.reshape(num_chunks, -1, 1, self.relation_dim)
                score = heads - tails
                return gamma - nd.norm(score, ord=1, axis=-1)
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

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

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

class ComplExScore(nn.Block):
    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = nd.split(edges.src['emb'], num_outputs=2, axis=-1)
        real_tail, img_tail = nd.split(edges.dst['emb'], num_outputs=2, axis=-1)
        real_rel, img_rel = nd.split(edges.data['emb'], num_outputs=2, axis=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': nd.sum(score, -1)}

    def prepare(self, g, gpu_id, trace=False):
        pass

    def create_neg_prepare(self, neg_head):
        def fn(rel_id, num_chunks, head, tail, gpu_id, trace=False):
            return head, tail
        return fn

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
                emb_real, emb_img = nd.split(tails, num_outputs=2, axis=-1)
                rel_real, rel_img = nd.split(relations, num_outputs=2, axis=-1)
                real = emb_real * rel_real + emb_img * rel_img
                img = -emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                heads = nd.transpose(heads, axes=(0, 2, 1))
                return nd.linalg_gemm2(tmp, heads)
            return fn
        else:
            def fn(heads, relations, tails, num_chunks, chunk_size, neg_sample_size):
                hidden_dim = heads.shape[1]
                emb_real, emb_img = nd.split(heads, num_outputs=2, axis=-1)
                rel_real, rel_img = nd.split(relations, num_outputs=2, axis=-1)
                real = emb_real * rel_real - emb_img * rel_img
                img = emb_real * rel_img + emb_img * rel_real
                emb_complex = nd.concat(real, img, dim=-1)
                tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)

                tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                tails = nd.transpose(tails, axes=(0, 2, 1))
                return nd.linalg_gemm2(tmp, tails)
            return fn
