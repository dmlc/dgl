import argparse
import scipy as sp
import dgl
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import ndarray as nd

from models.mxnet.score_fun import *

class ArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(ArgParser, self).__init__()

        self.add_argument('--score_func', default='ALL',
            choices=['ALL', 'TransE', 'DistMult'])

def allclose(a, b, rtol=1e-5, atol=1e-5):
    return np.allclose(a.asnumpy(),
            b.asnumpy(), rtol=rtol, atol=atol)

def transe_score(g):
    def edge_func(edges):
        gamma = 12.0
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head + rel - tail
        return {'score_1': gamma - nd.norm(score, ord=1, axis=-1)}

    g.apply_edges(lambda edges: edge_func(edges))

def distmult_score(g):
    def edge_func(edges):
        head = edges.src['emb']
        tail = edges.dst['emb']
        rel = edges.data['emb']
        score = head * rel * tail
        return {'score_1': nd.sum(score, axis=-1)}

    g.apply_edges(lambda edges: edge_func(edges))

def test_score_func(func_name):
    g = dgl.DGLGraph()
    g.add_nodes(30)
    for i in range(10):
        for j in range(20):
            g.add_edges(i, j + 10)

    ids = []
    for i in range(30):
        ids.append(i)

    g.ndata['id'] = nd.arange(30)
    g.edata['id'] = nd.from_numpy(np.array([0] * 200))
    nv = g.number_of_nodes()
    if func_name is 'TransE':
        ke_score_func = TransEScore(12.0)
        score_func = transe_score
    elif func_name is 'DistMult':
        ke_score_func = DistMultScore()
        score_func = distmult_score

    mx.random.seed(7)
    with mx.autograd.record():
        nd_emb = gluon.nn.Embedding(nv, 400)
        eg_emb = gluon.nn.Embedding(1, 400)
        nd_emb.initialize()
        eg_emb.initialize()
        g.ndata['emb'] = nd_emb(g.ndata['id'])
        g.edata['emb'] = eg_emb(g.edata['id'])

        ke_score_func(g)
        score_ke = g.edata.pop('score')
        score_ke = nd.log(nd.sigmoid(score_ke))
        score_ke = score_ke.mean()
        score_ke.backward()
        grad_ke = nd_emb.weight.grad()

    mx.random.seed(7)
    with mx.autograd.record():
        nd_emb = gluon.nn.Embedding(nv, 400)
        eg_emb = gluon.nn.Embedding(1, 400)
        nd_emb.initialize()
        eg_emb.initialize()
        g.ndata['emb'] = nd_emb(g.ndata['id'])
        g.edata['emb'] = eg_emb(g.edata['id'])

        score_func(g)
        score_udf = g.edata.pop('score_1')
        score_udf = nd.log(nd.sigmoid(score_udf))
        score_udf = score_udf.mean()
        score_udf.backward()
        grad_udf = nd_emb.weight.grad()

    assert(allclose(score_ke, score_udf))
    assert(allclose(grad_ke, grad_udf))

    mx.random.seed(7)
    with mx.autograd.record():
        nd_emb = gluon.nn.Embedding(nv, 400)
        eg_emb = gluon.nn.Embedding(1, 400)
        nd_emb.initialize()
        eg_emb.initialize()
        head = nd_emb(nd.arange(10))
        tail = nd_emb(nd.arange(start=10,stop=30))
        relation = eg_emb(nd.arange(1)).reshape((1, 400))
        neg_score = ke_score_func.create_neg(False, 1, 10, 20, 400)

        score_neg_tail = neg_score(head, relation, tail)
        score_neg_tail = nd.log(nd.sigmoid(score_neg_tail))
        score_neg_tail = score_neg_tail.mean()
        score_neg_tail.backward()
        grad_neg_tail = nd_emb.weight.grad()

    assert(allclose(score_neg_tail, score_udf))
    assert(allclose(grad_neg_tail, grad_udf))

    mx.random.seed(7)
    with mx.autograd.record():
        nd_emb = gluon.nn.Embedding(nv, 400)
        eg_emb = gluon.nn.Embedding(1, 400)
        nd_emb.initialize()
        eg_emb.initialize()
        head = nd_emb(nd.arange(10))
        tail = nd_emb(nd.arange(start=10,stop=30))
        relation = eg_emb(nd.arange(1)).reshape((1, 400))
        neg_score = ke_score_func.create_neg(True, 1, 20, 10, 400)

        score_neg_head = neg_score(head, relation, tail)
        score_neg_head = nd.log(nd.sigmoid(score_neg_head))
        score_neg_head = score_neg_head.mean()
        score_neg_head.backward()
        grad_neg_head = nd_emb.weight.grad()

    assert(allclose(score_neg_head, score_udf))
    assert(allclose(grad_neg_head, grad_udf))

def main(args):
    if args.score_func is 'ALL':
        for score_func in ['TransE', 'DistMult']:
            test_score_func(score_func)
    else:
        test_score_func(args.score_func)

if __name__ == '__main__':
    args = ArgParser().parse_args()

    main(args)
