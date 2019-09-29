import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as INIT

class TransEScore(nn.Module):
    def __init__(self, gamma):
        super(TransEScore, self).__init__()
        self.gamma = gamma

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def create_neg(self, neg_head, num_chunks, chunk_size, neg_sample_size, hidden_dim):
        gamma = self.gamma
        if neg_head:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                    tails = tails - relations
                    tails = tails.reshape(num_chunks, chunk_size, hidden_dim)
                    return gamma - th.cdist(tails, heads, p=1)
            return fn()
        else:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    heads = heads + relations
                    heads = heads.reshape(num_chunks, chunk_size, hidden_dim)
                    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                    return gamma - th.cdist(heads, tails, p=1)
            return fn()

class TransHScore(nn.Module):
    def __init__(self, gamma, n_relations, relation_dim, emb_init):
        super(TransHScore, self).__init__()
        self.norm_vector = nn.Embedding(n_relations, relation_dim)
        self.emb_init = emb_init
        self.gamma = gamma

    def edge_func(self, edges):
        def _transfer(e, norm):
            norm = F.normalize(norm, p=2, dim=-1)
            return e - th.sum(e * norm, -1, True) * norm

        rel = edges.data['emb']
        r_norm = edges.data['norm']
        head = _transfer(edges.src['emb'], r_norm)
        tail = _transfer(edges.dst['emb'], r_norm)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def reset_parameters(self):
        INIT.uniform_(self.norm_vector.weight, -self.emb_init, self.emb_init)

    def save(self, path, name):
        raise Exception('save norm vectors')

    def load(self, path, name):
        raise Exception('load norm vectors')

    def forward(self, g):
        g.edata['norm'] = self.norm_vector(g.edata['id'])
        g.apply_edges(lambda edges: self.edge_func(edges))

class TransRScore(nn.Module):
    def __init__(self, gamma, n_relations, relation_dim, entity_dim, emb_init):
        super(TransRScore, self).__init__()
        self.transfer_matrix = nn.Embedding(n_relations, relation_dim * entity_dim)
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        def _transfer(transfer_matrix, embeddings):
            """
            :param transfer_matrix: shape: batch, d_rel, d_ent
            :param embeddings:  shape: batch, d_ent, 1
            :return: shape: batch, d_rel, 1
            """
            return th.matmul(transfer_matrix, embeddings)

        head = edges.src['emb'].unsqueeze(-1)
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel_transfer = edges.data['transfer_mat'].view(rel.shape[0], self.relation_dim, self.entity_dim)
        head = _transfer(rel_transfer, head).squeeze(-1)
        tail = _transfer(rel_transfer, tail).squeeze(-1)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def save(self, path, name):
        raise Exception('save projection matrix')

    def load(self, path, name):
        raise Exception('load projection matrix')

    def reset_parameters(self):
        INIT.uniform_(self.transfer_matrix.weight, -self.emb_init, self.emb_init)

    def forward(self, g):
        g.edata['transfer_mat'] = self.transfer_matrix(g.edata['id'])
        g.apply_edges(lambda edges: self.edge_func(edges))

class TransDScore(nn.Module):
    def __init__(self, gamma, n_relations, relation_dim, n_entities, entity_dim, emb_init):
        super(TransDScore, self).__init__()
        self.ent_transfer = nn.Embedding(n_entities, entity_dim)
        self.rel_transfer = nn.Embedding(n_relations, relation_dim)
        self.gamma = gamma

    def edge_func(self, edges):
        def _transfer(e, e_transfer, r_transfer):
            e = e + th.sum(e * e_transfer, -1, True) * r_transfer
            e_norm = F.normalize(e, p=2, dim=-1)
            return e_norm

        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        head_t = edges.src['entity_t']
        tail_t = edges.dst['entity_t']
        rel_t = edges.data['rel_t']
        head = _transfer(head, head_t, rel_t)
        tail = _transfer(tail, tail_t, rel_t)
        score = head + rel - tail
        return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def save(self, path, name):
        raise Exception('save transfer')

    def load(self, path, name):
        raise Exception('load transfer')

    def reset_parameters(self):
        INIT.uniform_(self.ent_transfer.weight, -self.emb_init, self.emb_init)
        INIT.uniform_(self.rel_transfer.weight, -self.emb_init, self.emb_init)

    def forward(self, g):
        g.ndata['entity_t'] = self.ent_transfer(g.ndata['id'])
        g.edata['rel_t'] = self.rel_transfer(g.edata['id'])
        g.apply_edges(lambda edges: self.edge_func(edges))

class DistMultScore(nn.Module):
    def __init__(self):
        super(DistMultScore, self).__init__()

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, dim=-1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head, num_chunks, chunk_size, neg_sample_size, hidden_dim):
        if neg_head:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                    heads = th.transpose(heads, 1, 2)
                    tmp = (tails * relations).reshape(num_chunks, chunk_size, hidden_dim)
                    return th.bmm(tmp, heads)
            return fn()
        else:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                    tails = th.transpose(tails, 1, 2)
                    tmp = (heads * relations).reshape(num_chunks, chunk_size, hidden_dim)
                    return th.bmm(tmp, tails)
            return fn()

class ComplExScore(nn.Module):
    def __init__(self):
        super(ComplExScore, self).__init__()

    def edge_func(self, edges):
        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        real_rel, img_rel = th.chunk(edges.data['emb'], 2, dim=-1)

        score = real_head * real_tail * real_rel \
                + img_head * img_tail * real_rel \
                + real_head * img_tail * img_rel \
                - img_head * real_tail * img_rel
        # TODO: check if there exists minus sign and if gamma should be used here(jin)
        return {'score': th.sum(score, -1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

    def create_neg(self, neg_head, num_chunks, chunk_size, neg_sample_size, hidden_dim):
        if neg_head:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    emb_real = tails[..., :hidden_dim // 2]
                    emb_imag = tails[..., hidden_dim // 2:]
                    rel_real = relations[..., :hidden_dim // 2]
                    rel_imag = relations[..., hidden_dim // 2:]
                    real = emb_real * rel_real + emb_imag * rel_imag
                    imag = -emb_real * rel_imag + emb_imag * rel_real
                    emb_complex = th.cat((real, imag), dim=-1)
                    tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                    heads = heads.reshape(num_chunks, neg_sample_size, hidden_dim)
                    heads = th.transpose(heads, 1, 2)
                    return th.bmm(tmp, heads)
            return fn()
        else:
            class fn:
                def __init__(self):
                    self.num_chunks = num_chunks
                    self.chunk_size = chunk_size
                    self.neg_sample_size = neg_sample_size

                def __call__(self, heads, relations, tails):
                    emb_real = heads[..., :hidden_dim // 2]
                    emb_imag = heads[..., hidden_dim // 2:]
                    rel_real = relations[..., :hidden_dim // 2]
                    rel_imag = relations[..., hidden_dim // 2:]
                    real = emb_real * rel_real - emb_imag * rel_imag
                    imag = emb_real * rel_imag + emb_imag * rel_real
                    emb_complex = th.cat((real, imag), dim=-1)
                    tmp = emb_complex.reshape(num_chunks, chunk_size, hidden_dim)
                    tails = tails.reshape(num_chunks, neg_sample_size, hidden_dim)
                    tails = th.transpose(tails, 1, 2)
                    return th.bmm(tmp, tails)
            return fn()

class RotatEScore(nn.Module):
    def __init__(self, gamma, emb_init):
        super(RotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init

    def edge_func(self, edges):
        pi = 3.14159265358979323846

        real_head, img_head = th.chunk(edges.src['emb'], 2, dim=-1)
        real_tail, img_tail = th.chunk(edges.dst['emb'], 2, dim=-1)
        relation = edges.data['emb']

        phase_relation = relation / (self.emb_init / pi)

        real_rel, img_rel = th.cos(phase_relation), th.sin(phase_relation)

        real_score = real_head * real_rel - img_head * img_rel
        img_score = real_head * img_rel + img_head * real_rel
        real_score -= real_tail
        img_score -= img_tail

        score = th.stack([real_score, img_score], dim=0)
        score = th.norm(score, dim=0)
        score = self.gamma - score.sum(dim=-1)
        return {'score': score}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

class pRotatEScore(nn.Module):
    def __init__(self, gamma, emb_init):
        super(pRotatEScore, self).__init__()
        self.gamma = gamma
        self.emb_init = emb_init
        self.modulus = nn.Parameter(th.Tensor([[0.5 * self.emb_init]]))

    def edge_func(self, edges):
        pi = 3.14159262358979323846

        phase_head = edges.src['emb'] / (self.emb_init / pi)
        phase_tail = edges.dst['emb'] / (self.emb_init / pi)
        phase_rel = edges.data['emb'] / (self.emb_init / pi)

        score = (phase_head + phase_rel) - phase_tail
        score = score.sin().abs()
        score = (self.gamma - score.sum(dim=-1))
        score = score * self.modulus.item()
        return {'score': score}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        raise Exception('save modulus')

    def load(self, path, name):
        raise Exception('load modulus')

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))

class RESCALScore(nn.Module):
    def __init__(self, relation_dim, entity_dim):
        super(RESCALScore, self).__init__()
        self.relation_dim = relation_dim
        self.entity_dim = entity_dim

    def edge_func(self, edges):
        head = edges.src['emb']
        tail = edges.dst['emb'].unsqueeze(-1)
        rel = edges.data['emb']
        rel = rel.view(-1, self.relation_dim, self.entity_dim)
        score = head * th.matmul(rel, tail).squeeze(-1)
        # TODO: check if use self.gamma
        return {'score': th.sum(score, dim=-1)}
        # return {'score': self.gamma - th.norm(score, p=1, dim=-1)}

    def reset_parameters(self):
        pass

    def save(self, path, name):
        pass

    def load(self, path, name):
        pass

    def forward(self, g):
        g.apply_edges(lambda edges: self.edge_func(edges))
