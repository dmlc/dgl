import os
import scipy as sp
import dgl
import numpy as np
import dgl.backend as F
import dgl

backend = os.environ.get('DGLBACKEND')
if backend.lower() == 'mxnet':
    from models.mxnet.score_fun import *
else:
    from models.pytorch.score_fun import *
from models.general_models import KEModel
from dataloader.sampler import create_neg_subgraph

def generate_rand_graph(n, func_name):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = dgl.DGLGraph(arr, readonly=True)
    num_rels = 10
    entity_emb = F.uniform((g.number_of_nodes(), 10), F.float32, F.cpu(), 0, 1)
    rel_emb = F.uniform((num_rels, 10), F.float32, F.cpu(), 0, 1)
    if func_name == 'RESCAL':
        rel_emb = F.uniform((num_rels, 10*10), F.float32, F.cpu(), 0, 1)
    g.ndata['id'] = F.arange(0, g.number_of_nodes())
    rel_ids = np.random.randint(0, num_rels, g.number_of_edges(), dtype=np.int64)
    g.edata['id'] = F.tensor(rel_ids, F.int64)
    return g, entity_emb, rel_emb

ke_score_funcs = {'TransE': TransEScore(12.0),
                  'DistMult': DistMultScore(),
                  'ComplEx': ComplExScore(),
                  'RESCAL': RESCALScore(10, 10)}

class BaseKEModel:
    def __init__(self, score_func, entity_emb, rel_emb):
        self.score_func = score_func
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)
        self.entity_emb = entity_emb
        self.rel_emb = rel_emb

    def predict_score(self, g):
        g.ndata['emb'] = self.entity_emb[g.ndata['id']]
        g.edata['emb'] = self.rel_emb[g.edata['id']]
        self.score_func(g)
        return g.edata['score']

    def predict_neg_score(self, pos_g, neg_g):
        pos_g.ndata['emb'] = self.entity_emb[pos_g.ndata['id']]
        pos_g.edata['emb'] = self.rel_emb[pos_g.edata['id']]
        neg_g.ndata['emb'] = self.entity_emb[neg_g.ndata['id']]
        neg_g.edata['emb'] = self.rel_emb[neg_g.edata['id']]
        num_chunks = neg_g.num_chunks
        chunk_size = neg_g.chunk_size
        neg_sample_size = neg_g.neg_sample_size
        if neg_g.neg_head:
            neg_head_ids = neg_g.ndata['id'][neg_g.head_nid]
            neg_head = self.entity_emb[neg_head_ids]
            _, tail_ids = pos_g.all_edges(order='eid')
            tail = pos_g.ndata['emb'][tail_ids]
            rel = pos_g.edata['emb']
            neg_score = self.head_neg_score(neg_head, rel, tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_emb[neg_tail_ids]
            head_ids, _ = pos_g.all_edges(order='eid')
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        return neg_score

def check_score_func(func_name):
    batch_size = 10
    neg_sample_size = 10
    g, entity_emb, rel_emb = generate_rand_graph(100, func_name)
    hidden_dim = entity_emb.shape[1]
    ke_score_func = ke_score_funcs[func_name]
    model = BaseKEModel(ke_score_func, entity_emb, rel_emb)

    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
    sampler = EdgeSampler(g, batch_size=batch_size,
                          neg_sample_size=neg_sample_size,
                          negative_mode='PBG-head',
                          num_workers=1,
                          shuffle=False,
                          exclude_positive=False,
                          return_false_neg=False)

    for pos_g, neg_g in sampler:
        neg_g = create_neg_subgraph(pos_g, neg_g, True, True, g.number_of_nodes())
        pos_g.copy_from_parent()
        neg_g.copy_from_parent()
        score1 = F.reshape(model.predict_score(neg_g), (batch_size, -1))
        score2 = model.predict_neg_score(pos_g, neg_g)
        score2 = F.reshape(score2, (batch_size, -1))
        np.testing.assert_allclose(F.asnumpy(score1), F.asnumpy(score2),
                                   rtol=1e-5, atol=1e-5)

def test_score_func():
    for key in ke_score_funcs:
        check_score_func(key)

if __name__ == '__main__':
    test_score_func()
