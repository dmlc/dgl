import os
import scipy as sp
import dgl
import numpy as np
import dgl.backend as F
import dgl

backend = os.environ.get('DGLBACKEND', 'pytorch')
if backend.lower() == 'mxnet':
    import mxnet as mx
    mx.random.seed(42)
    np.random.seed(42)

    from models.mxnet.score_fun import *
    from models.mxnet.tensor_models import ExternalEmbedding
else:
    import torch as th
    th.manual_seed(42)
    np.random.seed(42)

    from models.pytorch.score_fun import *
    from models.pytorch.tensor_models import ExternalEmbedding
from models.general_models import KEModel
from dataloader.sampler import create_neg_subgraph

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def generate_rand_graph(n, func_name):
    arr = (sp.sparse.random(n, n, density=0.1, format='coo') != 0).astype(np.int64)
    g = dgl.DGLGraph(arr, readonly=True)
    num_rels = 10
    entity_emb = F.uniform((g.number_of_nodes(), 10), F.float32, F.cpu(), 0, 1)
    if func_name == 'RotatE':
        entity_emb = F.uniform((g.number_of_nodes(), 20), F.float32, F.cpu(), 0, 1)
    rel_emb = F.uniform((num_rels, 10), F.float32, F.cpu(), -1, 1)
    if func_name == 'RESCAL':
        rel_emb = F.uniform((num_rels, 10*10), F.float32, F.cpu(), 0, 1)
    g.ndata['id'] = F.arange(0, g.number_of_nodes())
    rel_ids = np.random.randint(0, num_rels, g.number_of_edges(), dtype=np.int64)
    g.edata['id'] = F.tensor(rel_ids, F.int64)
    # TransR have additional projection_emb
    if (func_name == 'TransR'):
        args = {'gpu':-1, 'lr':0.1}
        args = dotdict(args)
        projection_emb = ExternalEmbedding(args, 10, 10 * 10, F.cpu())
        return g, entity_emb, rel_emb, (12.0, projection_emb, 10, 10)
    elif (func_name == 'TransE'):
        return g, entity_emb, rel_emb, (12.0)
    elif (func_name == 'TransE_l1'):
        return g, entity_emb, rel_emb, (12.0, 'l1')
    elif (func_name == 'TransE_l2'):
        return g, entity_emb, rel_emb, (12.0, 'l2')
    elif (func_name == 'RESCAL'):
        return g, entity_emb, rel_emb, (10, 10)
    elif (func_name == 'RotatE'):
        return g, entity_emb, rel_emb, (12.0, 1.0)
    else:
        return g, entity_emb, rel_emb, None

ke_score_funcs = {'TransE': TransEScore,
                  'TransE_l1': TransEScore,
                  'TransE_l2': TransEScore,
                  'DistMult': DistMultScore,
                  'ComplEx': ComplExScore,
                  'RESCAL': RESCALScore,
                  'TransR': TransRScore,
                  'RotatE': RotatEScore}

class BaseKEModel:
    def __init__(self, score_func, entity_emb, rel_emb):
        self.score_func = score_func
        self.head_neg_score = self.score_func.create_neg(True)
        self.tail_neg_score = self.score_func.create_neg(False)
        self.head_neg_prepare = self.score_func.create_neg_prepare(True)
        self.tail_neg_prepare = self.score_func.create_neg_prepare(False)
        self.entity_emb = entity_emb
        self.rel_emb = rel_emb
        # init score_func specific data if needed
        self.score_func.reset_parameters()

    def predict_score(self, g):
        g.ndata['emb'] = self.entity_emb[g.ndata['id']]
        g.edata['emb'] = self.rel_emb[g.edata['id']]
        self.score_func.prepare(g, -1, False)
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

            neg_head, tail = self.head_neg_prepare(pos_g.edata['id'], num_chunks, neg_head, tail, -1, False)
            neg_score = self.head_neg_score(neg_head, rel, tail,
                                            num_chunks, chunk_size, neg_sample_size)
        else:
            neg_tail_ids = neg_g.ndata['id'][neg_g.tail_nid]
            neg_tail = self.entity_emb[neg_tail_ids]
            head_ids, _ = pos_g.all_edges(order='eid')
            head = pos_g.ndata['emb'][head_ids]
            rel = pos_g.edata['emb']

            head, neg_tail = self.tail_neg_prepare(pos_g.edata['id'], num_chunks, head, neg_tail, -1, False)
            neg_score = self.tail_neg_score(head, rel, neg_tail,
                                            num_chunks, chunk_size, neg_sample_size)

        return neg_score

def check_score_func(func_name):
    batch_size = 10
    neg_sample_size = 10
    g, entity_emb, rel_emb, args = generate_rand_graph(100, func_name)
    hidden_dim = entity_emb.shape[1]
    ke_score_func = ke_score_funcs[func_name]
    if args is None:
        ke_score_func = ke_score_func()
    elif type(args) is tuple:
        ke_score_func = ke_score_func(*list(args))
    else:
        ke_score_func = ke_score_func(args)
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

def test_score_func_transe():
    check_score_func('TransE')
    check_score_func('TransE_l1')
    check_score_func('TransE_l2')

def test_score_func_distmult():
    check_score_func('DistMult')

def test_score_func_complex():
    check_score_func('ComplEx')

def test_score_func_rescal():
    check_score_func('RESCAL')

def test_score_func_transr():
    check_score_func('TransR')

def test_score_func_rotate():
    check_score_func('RotatE')
        
if __name__ == '__main__':
    test_score_func_transe()
    test_score_func_distmult()
    test_score_func_complex()
    test_score_func_rescal()
    test_score_func_transr()
    test_score_func_rotate()
