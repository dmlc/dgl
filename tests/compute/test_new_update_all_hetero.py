import dgl
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
        ('user', 'follows', 'user'):  ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'plays', 'game'): ([0, 1, 2, 1], [0, 0, 1, 1]),
        ('user', 'wishes', 'game'): ([0, 1, 1], [0, 0, 1]),
        ('developer', 'develops', 'game'): ([0, 1, 0], [0, 1, 1]),
    }, idtype=idtype, device=F.ctx())
    assert g.idtype == idtype
    assert g.device == F.ctx()
    return g

# def init_features(idtype):


@parametrize_dtype
def test_unary_copy_u(idtype):
    def _test(mfunc, rfunc):

        g = create_test_heterograph(idtype)

        x1 = F.randn((g.num_nodes('user'), feat_size))
        x2 = F.randn((g.num_nodes('developer'), feat_size))

        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes['user'].data['h'] = x1
        g.nodes['developer'].data['h'] = x2

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {'plays' : (mfunc('h', 'm'), rfunc('m', 'y')),
                'follows': (mfunc('h', 'm'), rfunc('m', 'y')),
                'develops': (mfunc('h', 'm'), rfunc('m', 'y')),
                'wishes': (mfunc('h', 'm'), rfunc('m', 'y'))},
                'sum')
            r1 = g.nodes['game'].data['y']
            F.backward(r1, F.randn(r1.shape))
            n_grad1 = F.grad(g.nodes['user'].data['h'])
            g.nodes['game'].data.clear()

        #################################################################
        #  update_all(): call msg_passing for all etypes
        #################################################################

        g.update_all(mfunc('h', 'm'), rfunc('m', 'y'))
        r2 = g.nodes['game'].data['y']
        F.backward(r2, F.randn(r2.shape))
        n_grad2 = F.grad(g.nodes['user'].data['h'])

        # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        if not F.allclose(n_grad1, n_grad2):
            print('node grad')
            _print_error(n_grad1, n_grad2)
        assert(F.allclose(n_grad1, n_grad2))

    _test(fn.copy_u, fn.sum)
    # TODO(Israt) :Add reduce func to suport the following reduce op
    # _test('copy_u', 'max')
    # _test('copy_u', 'min')
    # _test('copy_u', 'mean')

@parametrize_dtype
def test_unary_copy_e(idtype):
    def _test(mfunc, rfunc):

        g = create_test_heterograph(idtype)
        feat_size = 2

        x1 = F.randn((4,feat_size))
        x2 = F.randn((4,feat_size))
        x3 = F.randn((3,feat_size))
        x4 = F.randn((3,feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g['plays'].edata['eid'] = x1
        g['follows'].edata['eid'] = x2
        g['develops'].edata['eid'] = x3
        g['wishes'].edata['eid'] = x4

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {'plays' : (mfunc('eid', 'm'), rfunc('m', 'y')),
                'follows': (mfunc('eid', 'm'), rfunc('m', 'y')),
                'develops': (mfunc('eid', 'm'), rfunc('m', 'y')),
                'wishes': (mfunc('eid', 'm'), rfunc('m', 'y'))},
                'sum')
            r1 = g.nodes['game'].data['y']
            F.backward(r1, F.randn(r1.shape))
            e_grad1 = F.grad(g['develops'].edata['eid'])


        #################################################################
        #  update_all(): call msg_passing for all etypes
        #################################################################

        # TODO(Israt): output type can be None in multi_update and empty
        # tensor in new_update_all
        g.update_all(mfunc('eid', 'm'), rfunc('m', 'y'))
        r2 = g.nodes['game'].data['y']
        F.backward(r2, F.randn(r2.shape))
        e_grad2 = F.grad(g['develops'].edata['eid'])

        # # correctness check
        def _print_error(a, b):
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                   print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        if not F.allclose(e_grad1, e_grad2):
            print('edge grad')
            _print_error(e_grad1, e_grad2)
        assert(F.allclose(e_grad1, e_grad2))

    _test(fn.copy_e, fn.sum)
    # TODO(Israt) :Add reduce func to suport the following reduce op
    # _test('copy_e', 'max')
    # _test('copy_e', 'min')
    # _test('copy_e', 'mean')


if __name__ == '__main__':
    test_unary_copy_u()
    test_unary_copy_e()


