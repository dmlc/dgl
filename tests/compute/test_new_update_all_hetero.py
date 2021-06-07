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
import torch

rfuncs = {'sum': fn.sum, 'max': fn.max, 'min': fn.min, 'mean': fn.mean}
fill_value = {'sum': 0, 'max': float("-inf")}
    # g.nodes['user'].data['h'] = F.randn((3, 2))
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

@parametrize_dtype
def test_unary_copy_u(idtype):
    def _test(mfunc, rfunc):

        g = create_test_heterograph(idtype)
        feat_size = 2

        x1 = F.ones((g.num_nodes('user'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x2 = F.ones((g.num_nodes('developer'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x3 = F.ones((g.num_nodes('game'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)

        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes['user'].data['h'] = x1
        g.nodes['developer'].data['h'] = x2 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)
        g.nodes['game'].data['h'] = x3 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)

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
            F.backward(r1, F.ones(r1.shape))
            n_grad1 = F.grad(g.nodes['user'].data['h'])
            g.nodes['game'].data.clear()

        #################################################################
        #  update_all_new(): call msg_passing for all etypes
        #################################################################

        g.update_all_new(mfunc('h', 'm'), rfunc('m', 'y'))
        r2 = g.nodes['game'].data['y']
        F.backward(r2, F.ones(r2.shape))
        n_grad2 = F.grad(g.nodes['user'].data['h'])

        # correctness check
        def _print_error(a, b):
            print("ERROR: Test copy_src_{} partial: {}".
              format(red, partial))
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
    # _test('copy_u', 'max')
    # _test('copy_u', 'min')
    # _test('copy_u', 'mean')

@parametrize_dtype
def test_unary_copy_e(idtype):
    def _test(mfunc, rfunc):

        g = create_test_heterograph(idtype)
        feat_size = 2

        x1 = F.ones((g.num_nodes('user'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x2 = F.ones((g.num_nodes('developer'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x3 = F.ones((g.num_nodes('game'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)

        # x2 = torch.full((g.num_nodes('user'), feat_size) , 3.0)
        F.attach_grad(x1)
        F.attach_grad(x2)
        g.nodes['user'].data['h'] = x1
        g.nodes['developer'].data['h'] = x2 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)
        g.nodes['game'].data['h'] = x3 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)

        x1 = F.ones((4,feat_size))
        x2 = F.ones((4,feat_size))
        x3 = F.ones((3,feat_size))
        x4 = F.ones((3,feat_size))
        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        F.attach_grad(x4)
        g['plays'].edata['eid'] = x1 #torch.full((4, 2), 5.0)
        g['follows'].edata['eid'] = x2  #torch.full((4, 2), 10.0)
        g['develops'].edata['eid'] = x3 #F.ones((3,feat_size)) # torch.full((3, 2), 20.0)
        g['wishes'].edata['eid'] = x4 #F.ones((3,feat_size)) #torch.full((3, 2), 15.0)

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
            F.backward(r1, F.ones(r1.shape))
            e_grad1 = F.grad(g['follows'].edata['eid'])


        #################################################################
        #  update_all_new(): call msg_passing for all etypes
        #################################################################

        g.update_all_new(mfunc('eid', 'm'), rfunc('m', 'y'))
        r2 = g.nodes['game'].data['y']
        # TODO (Israt): fix backward
        # F.backward(r2, F.ones(r2.shape))
        # e_grad2 = F.grad(g['follows'].edata['eid'])

        # correctness check
        def _print_error(a, b):
            print("ERROR: Test copy_src_{} partial: {}".
              format(red, partial))
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        # if not F.allclose(n_grad1, n_grad2):
        #     print('node grad')
        #     _print_error(n_grad1, n_grad2)
        # assert(F.allclose(n_grad1, n_grad2))

    _test(fn.copy_e, fn.sum)
    # _test('copy_e', 'max')
    # _test('copy_e', 'min')
    # _test('copy_e', 'mean')


@parametrize_dtype
def test_binary_u_op_v(idtype):
    def _test(mfunc, rfunc):

        g = create_test_heterograph(idtype)
        feat_size = 2

        x1 = F.ones((g.num_nodes('user'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x2 = F.ones((g.num_nodes('developer'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)
        x3 = F.ones((g.num_nodes('game'), feat_size)) #torch.full((g.num_nodes('game'), feat_size), 2.0)

        F.attach_grad(x1)
        F.attach_grad(x2)
        F.attach_grad(x3)
        g.nodes['user'].data['h'] = x1
        g.nodes['developer'].data['h'] = x2 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)
        g.nodes['game'].data['h'] = x3 #torch.full((g.num_nodes('developer'), feat_size) ,5.0)

        #################################################################
        #  multi_update_all(): call msg_passing separately for each etype
        #################################################################

        with F.record_grad():
            g.multi_update_all(
                {'plays' : (mfunc('h', 'h', 'm'), rfunc('m', 'y')),
                'follows': (mfunc('h', 'h', 'm'), rfunc('m', 'y')),
                'develops': (mfunc('h', 'h', 'm'), rfunc('m', 'y')),
                'wishes': (mfunc('h', 'h', 'm'), rfunc('m', 'y'))},
                'sum')
            r1 = g.nodes['game'].data['y']
            F.backward(r1, F.ones(r1.shape))
            n_grad1 = F.grad(g.nodes['user'].data['h'])

        #################################################################
        #  update_all_new(): call msg_passing for all etypes
        #################################################################

        g.update_all_new(mfunc('h', 'h', 'm'), rfunc('m', 'y'))
        r2 = g.nodes['game'].data['y']
        # F.backward(r2, F.ones(r2.shape))
        # n_grad2 = F.grad(g.nodes['user'].data['h'])

        # correctness check
        def _print_error(a, b):
            print("ERROR: Test copy_src_{} partial: {}".
              format(red, partial))
            for i, (x, y) in enumerate(zip(F.asnumpy(a).flatten(), F.asnumpy(b).flatten())):
                if not np.allclose(x, y):
                    print('@{} {} v.s. {}'.format(i, x, y))

        if not F.allclose(r1, r2):
            _print_error(r1, r2)
        assert F.allclose(r1, r2)
        # if not F.allclose(n_grad1, n_grad2):
        #     print('node grad')
        #     _print_error(n_grad1, n_grad2)
        # assert(F.allclose(n_grad1, n_grad2))

    for binary_op in [fn.u_add_v, fn.u_sub_v, fn.u_mul_v]: #, "div"]:
        for reducer in [fn.sum]: #, "max", "min", "mean"]:
            print(binary_op, reducer)
            _test(binary_op, reducer)


if __name__ == '__main__':
    test_unary_copy_u()
    test_unary_copy_e()
    test_binary_u_op_v()

