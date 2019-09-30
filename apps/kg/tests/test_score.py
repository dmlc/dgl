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
from models.general_models import BaseKEModel

def generate_rand_graph(n):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = dgl.DGLGraph(arr, readonly=True)
    num_rels = 10
    entity_emb = F.uniform((g.number_of_nodes(), 10), F.float32, F.cpu(), 0, 1)
    rel_emb = F.uniform((num_rels, 10), F.float32, F.cpu(), 0, 1)
    g.ndata['id'] = F.arange(0, g.number_of_nodes())
    rel_ids = np.random.randint(0, num_rels, g.number_of_edges(), dtype=np.int64)
    g.edata['id'] = F.tensor(rel_ids, F.int64)
    return g, entity_emb, rel_emb

ke_score_funcs = {'TransE': TransEScore(12.0),
                  'DistMult': DistMultScore()}

class Embedding:
    def __init__(self, arr):
        self.arr = arr

    def __call__(self, idx, gpu_id=-1, trace=True):
        return self.arr[idx]

def check_score_func(func_name):
    batch_size = 10
    neg_sample_size = 10
    g, entity_emb, rel_emb = generate_rand_graph(100)
    hidden_dim = entity_emb.shape[1]
    ke_score_func = ke_score_funcs[func_name]
    entity_emb = Embedding(entity_emb)
    model = BaseKEModel(ke_score_func, batch_size, neg_sample_size,
                        hidden_dim, g.number_of_nodes())

    EdgeSampler = getattr(dgl.contrib.sampling, 'EdgeSampler')
    sampler = EdgeSampler(g, batch_size=batch_size,
                          neg_sample_size=neg_sample_size,
                          negative_mode='PBG-head',
                          num_workers=1,
                          shuffle=False,
                          exclude_positive=False,
                          return_false_neg=False)

    for pos_g, neg_g in sampler:
        pos_g.copy_from_parent()
        neg_g.copy_from_parent()
        pos_g.ndata['emb'] = entity_emb(pos_g.ndata['id'])
        pos_g.edata['emb'] = rel_emb[pos_g.edata['id']]
        neg_g.ndata['emb'] = entity_emb(neg_g.ndata['id'])
        neg_g.edata['emb'] = rel_emb[neg_g.edata['id']]
        score1 = F.reshape(model.predict_score(neg_g), (batch_size, -1))
        score2 = model.predict_neg_score(pos_g, neg_g, entity_emb, True)
        score2 = F.reshape(score2, (batch_size, -1))
        np.testing.assert_allclose(F.asnumpy(score1), F.asnumpy(score2),
                                   rtol=1e-5, atol=1e-5)

def test_score_func():
    for key in ke_score_funcs:
        check_score_func(key)
    
if __name__ == '__main__':
    test_score_func()
