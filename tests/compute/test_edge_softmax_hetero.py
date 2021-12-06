import dgl
from dgl.ops import edge_softmax
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx
import unittest, pytest
from dgl import DGLError
import test_utils
from test_utils import parametrize_dtype, get_cases
from scipy.sparse import rand
# import torch

rfuncs = {'sum': fn.sum, 'max': fn.max, 'min': fn.min, 'mean': fn.mean}
fill_value = {'sum': 0, 'max': float("-inf")}
feat_size = 2

@unittest.skipIf(dgl.backend.backend_name != 'pytorch', reason='Only support PyTorch for now')


def create_test_heterograph(idtype):
    # test heterograph from the docstring, plus a user -- wishes -- game relation
    # 3 users, 2 games, 2 developers
    # metagraph:
    #    ('user', 'follows', 'user'),
    #    ('user', 'plays', 'game'),
    #    ('user', 'wishes', 'game'),
    #    ('developer', 'develops', 'game')])

    g = dgl.heterograph({
        ('user', 'follows', 'user'):  ([0, 1, 2, 1, 1], [0, 0, 1, 1, 2]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 1, 1], [0, 0, 1]),
        ('developer', 'develops', 'game'): ([0, 1, 0], [0, 1, 1]),
    }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g


@pytest.mark.parametrize('g', get_cases(['clique']))
@pytest.mark.parametrize('norm_by', ['dst']) #, 'dst']) #''src
# @pytest.mark.parametrize('shp', edge_softmax_shapes)
@parametrize_dtype
def test_edge_softmax(g, norm_by, idtype):
    print("params", norm_by, idtype)

    g = create_test_heterograph(idtype)
    # x1 = torch.full((g.num_edges('follows'), feat_size), 2.0)
    # x2 = torch.full((g.num_edges('plays'), feat_size), 3.0)
    # x3 = torch.full((g.num_edges('wishes'), feat_size), 4.0)
    # x4 = torch.full((g.num_edges('develops'), feat_size), 5.0)

    x1 = F.randn((g.num_edges('follows'),feat_size))
    x2 = F.randn((g.num_edges('plays'),feat_size))
    x3 = F.randn((g.num_edges('wishes'),feat_size))
    x4 = F.randn((g.num_edges('develops'),feat_size))

    e1 = F.attach_grad(F.clone(x1))
    e2 = F.attach_grad(F.clone(x2))
    e3 = F.attach_grad(F.clone(x3))
    e4 = F.attach_grad(F.clone(x4))

    #################################################################
    #  edge_softmax() on homogeneous graph
    #################################################################

    with F.record_grad():
        score1 = edge_softmax(g['follows'], e1, norm_by=norm_by)
        score2 = edge_softmax(g['plays'], e2, norm_by=norm_by)
        score3 = edge_softmax(g['wishes'], e3, norm_by=norm_by)
        score4 = edge_softmax(g['develops'], e4, norm_by=norm_by)

        F.backward(F.reduce_sum(score1))
        grad_edata = F.grad(e1)

    #################################################################
    #  edge_softmax() on heterogeneous graph
    #################################################################

    e = {('user', 'follows', 'user'): e1,
        ('user', 'plays', 'game'): e2,
        ('user', 'wishes', 'game'): e3,
        ('developer', 'develops', 'game'): e4}
    # e['follows'] = F.attach_grad(F.clone(x1))
    # e['plays'] = F.attach_grad(F.clone(x2))
    # e['wishes'] = F.attach_grad(F.clone(x3))
    # e['develops'] = F.attach_grad(F.clone(x4))

    with F.record_grad():
        score = edge_softmax(g, e, norm_by=norm_by)
        loss = score[0].sum()
        F.backward(F.reduce_sum(score[0]))
        grad_edata = F.grad(e1)

        # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

if __name__ == '__main__':
    test_edge_softmax()

