from dgl.ops import gspmm, gsddmm
from utils import parametrize_dtype
import dgl
import random
import pytest
import networkx as nx
import backend as F
import numpy as np 
from utils import parametrize_dtype

random.seed(42)
np.random.seed(42)

udf_msg = {
    'add': lambda edges: {'m': edges.src['x'] + edges.data['w']},
    'sub': lambda edges: {'m': edges.src['x'] - edges.data['w']},
    'mul': lambda edges: {'m': edges.src['x'] * edges.data['w']},
    'div': lambda edges: {'m': edges.src['x'] / edges.data['w']},
    'copy_lhs': lambda edges: {'m': edges.src['x']},
    'copy_rhs': lambda edges: {'m': edges.data['w']}
}

def select(target, src, edge, dst):
    if target == 'u':
        return src
    elif target == 'v':
        return dst
    elif target == 'e':
        return edge

def binary_op(msg, x, y):
    if msg == 'add':
        return x + y
    elif msg == 'sub':
        return x - y
    elif msg == 'mul':
        return x * y
    elif msg == 'div':
        return x / y
    elif msg == 'dot':
        return F.sum(x * y, -1, keepdims=True)
    elif msg == 'copy_lhs':
        return x
    elif msg == 'copy_rhs':
        return y

def edge_func(lhs_target, rhs_target, msg):
    def foo(edges):
        return {
            'm': binary_op(
                msg,
                select(lhs_target, edges.src, edges.data, edges.dst)['x'],
                select(rhs_target, edges.src, edges.data, edges.dst)['y']
            )
        }
    return foo

udf_apply_edges = {
    lhs_target + '_' + msg + '_' + rhs_target: edge_func(lhs_target, rhs_target, msg)
    for lhs_target in ['u', 'v', 'e']
    for rhs_target in ['u', 'v', 'e']
    for msg in ['add', 'sub', 'mul', 'div', 'dot', 'copy_lhs', 'copy_rhs']
}

udf_reduce = {
    'sum': lambda nodes: {'v': F.sum(nodes.mailbox['m'], 1)},
    'min': lambda nodes: {'v': F.min(nodes.mailbox['m'], 1)},
    'max': lambda nodes: {'v': F.max(nodes.mailbox['m'], 1)}
}

graphs = [
#    dgl.rand_graph(30, 0),
    dgl.rand_graph(30, 100),
    dgl.rand_bipartite(30, 40, 300)
]

spmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((3, 3), (1, 3)),
    ((1,), (3,)),
    ((3,), (1,)),
    ((1,), (1,))
]

sddmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((5, 3, 1, 7), (1, 3, 7, 7)),
    ((1, 3, 3), (4, 1, 3)),
    ((3,), (3,)),
    ((1,), (1,))
]

@pytest.mark.parametrize('g', graphs)
@pytest.mark.parametrize('shp', spmm_shapes)
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'copy_lhs', 'copy_rhs'])
@pytest.mark.parametrize('reducer', ['sum', 'min', 'max'])
@parametrize_dtype
def test_spmm(idtype, g, shp, msg, reducer):
    g = g.astype(idtype).to(F.ctx())
    print(g)
    print(g.idtype)

    hu = F.tensor(np.random.rand(*((g.number_of_src_nodes(),) + shp[0])) + 1)
    he = F.tensor(np.random.rand(*((g.number_of_edges(),) + shp[1])) + 1)
    print('u shape: {}, e shape: {}'.format(F.shape(hu), F.shape(he)))

    g.srcdata['x'] = F.attach_grad(F.clone(hu))
    g.edata['w'] = F.attach_grad(F.clone(he))
    print('SpMM(message func: {}, reduce func: {})'.format(msg, reducer))

    u = F.attach_grad(F.clone(hu))
    e = F.attach_grad(F.clone(he))
    with F.record_grad():
        v = gspmm(g, msg, reducer, u, e)
        non_degree_indices = F.tensor(
            np.nonzero(F.asnumpy(g.in_degrees()) != 0)[0])
        v = F.gather_row(v, non_degree_indices)
        if g.number_of_edges() > 0:
            F.backward(F.reduce_sum(v))
            if msg != 'copy_rhs':
                grad_u = F.grad(u)
            if msg != 'copy_lhs':
                grad_e = F.grad(e)

    with F.record_grad():
        g.update_all(udf_msg[msg], udf_reduce[reducer])
        if g.number_of_edges() > 0:
            v1 = F.gather_row(g.dstdata['v'], non_degree_indices)
            assert F.allclose(v, v1)
            print('forward passed')

            F.backward(F.reduce_sum(v1))
            if msg != 'copy_rhs':
                if reducer in ['min', 'max']: # there might be some numerical errors
                    rate = F.reduce_sum(F.abs(F.grad(g.srcdata['x']) - grad_u)) /\
                           F.reduce_sum(F.abs(grad_u))
                    assert F.as_scalar(rate) < 1e-2, rate
                else:
                    assert F.allclose(F.grad(g.srcdata['x']), grad_u)
            if msg != 'copy_lhs':
                if reducer in ['min', 'max']:
                    rate = F.reduce_sum(F.abs(F.grad(g.edata['w']) - grad_e)) /\
                           F.reduce_sum(F.abs(grad_e))
                    assert F.as_scalar(rate) < 1e-2, rate
                else:
                    assert F.allclose(F.grad(g.edata['w']), grad_e)
            print('backward passed')

    g.srcdata.pop('x')
    g.edata.pop('w')
    if 'v' in g.dstdata: g.dstdata.pop('v')

@pytest.mark.parametrize('g', graphs)
@pytest.mark.parametrize('shp', sddmm_shapes)
@pytest.mark.parametrize('lhs_target', ['u', 'v', 'e'])
@pytest.mark.parametrize('rhs_target', ['u', 'v', 'e'])
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'dot', 'copy_lhs', 'copy_rhs'])
@parametrize_dtype
def test_sddmm(g, shp, lhs_target, rhs_target, msg, idtype):
    if lhs_target == rhs_target:
        return
    g = g.astype(idtype).to(F.ctx())
    if dgl.backend.backend_name == 'mxnet' and g.number_of_edges() == 0:
        pytest.skip()   # mxnet do not support zero shape tensor
    print(g)
    print(g.idtype)

    len_lhs = select(
        lhs_target,
        g.number_of_src_nodes(),
        g.number_of_edges(),
        g.number_of_dst_nodes())
    lhs_shp = (len_lhs,) + shp[0]
    len_rhs = select(
        rhs_target,
        g.number_of_src_nodes(),
        g.number_of_edges(),
        g.number_of_dst_nodes())
    rhs_shp = (len_rhs,) + shp[1]
    feat_lhs = F.tensor(np.random.rand(*lhs_shp) + 1)
    feat_rhs = F.tensor(np.random.rand(*rhs_shp) + 1)
    print('lhs shape: {}, rhs shape: {}'.format(F.shape(feat_lhs), F.shape(feat_rhs)))

    lhs_frame = select(
        lhs_target,
        g.srcdata,
        g.edata,
        g.dstdata)
    rhs_frame = select(
        rhs_target,
        g.srcdata,
        g.edata,
        g.dstdata)
    lhs_frame['x'] = F.attach_grad(F.clone(feat_lhs))
    rhs_frame['y'] = F.attach_grad(F.clone(feat_rhs))
    msg_func = lhs_target + '_' + msg + '_' + rhs_target
    print('SDDMM(message func: {})'.format(msg_func))

    lhs = F.attach_grad(F.clone(feat_lhs))
    rhs = F.attach_grad(F.clone(feat_rhs))
    with F.record_grad():
        e = gsddmm(g, msg, lhs, rhs, lhs_target=lhs_target, rhs_target=rhs_target)
        F.backward(F.reduce_sum(e))
        grad_lhs = F.grad(lhs)
        grad_rhs = F.grad(rhs)

    with F.record_grad():
        g.apply_edges(udf_apply_edges[msg_func])
        if g.number_of_edges() > 0:
            e1 = g.edata['m']
            assert F.allclose(e, e1)
            print('forward passed')

            F.backward(F.reduce_sum(e1))
            if msg != 'copy_rhs':
                assert F.allclose(F.grad(lhs_frame['x']), grad_lhs)
            if msg != 'copy_lhs':
                assert F.allclose(F.grad(rhs_frame['y']), grad_rhs)
            print('backward passed')

    lhs_frame.pop('x')
    rhs_frame.pop('y')
    if 'm' in g.edata: g.edata.pop('m')

if __name__ == '__main__':
    test_spmm(F.int32, graphs[0], spmm_shapes[5], 'copy_lhs', 'sum')
