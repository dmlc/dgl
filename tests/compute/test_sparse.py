import dgl
import dgl.sparse
import pytest
import networkx as nx
import backend as F
import numpy as np

np.random.seed(42)
dgl.random.seed(42)

def _unsqueeze_if_scalar(x):    # used in udf, to unsqueeze the feature if it's scalar
    return x if F.ndim(x) > 1 else F.unsqueeze(x, -1)

udf_msg = {
    'add': lambda edges: {'m': edges.src['x'] + edges.data['w']},
    'sub': lambda edges: {'m': edges.src['x'] - edges.data['w']},
    'mul': lambda edges: {'m': edges.src['x'] * edges.data['w']},
    'div': lambda edges: {'m': edges.src['x'] / edges.data['w']},
    'copy_u': lambda edges: {'m': edges.src['x']},
    'copy_e': lambda edges: {'m': edges.data['w']}
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
    dgl.rand_graph(30, 0),
    dgl.rand_graph(100, 30),
    dgl.rand_graph(100, 3000),
    dgl.rand_bipartite(80, 160, 3000)
]

spmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((5, 3, 1, 7), (1, 3, 7, 1)),
    ((1, 3, 1), (4, 1, 3)),
    ((3, 3), (1, 3)),
    ((), (3,)),
    ((3,), ()),
    ((), ())
]

sddmm_shapes = [
    ((1, 2, 1, 3, 1), (4, 1, 3, 1, 1)),
    ((5, 3, 1, 7), (1, 3, 7, 7)),
    ((1, 3, 3), (4, 1, 3)),
    ((3, 3), (1, 3)),
    ((3,), (3,)),
    ((), ())
]

@pytest.mark.parametrize('g', graphs)
@pytest.mark.parametrize('shp', spmm_shapes)
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'copy_u', 'copy_e'])
@pytest.mark.parametrize('reducer', ['sum', 'min', 'max'])
def test_spmm(g, shp, msg, reducer):
    print(g)
    u = F.tensor(np.random.rand(*((g.number_of_src_nodes(),) + shp[0])) + 1)
    e = F.tensor(np.random.rand(*((g.number_of_edges(),) + shp[1])) + 1)
    print('u shape: {}, e shape: {}'.format(F.shape(u), F.shape(e)))
    g.srcdata['x'] = _unsqueeze_if_scalar(u)
    g.edata['w'] = _unsqueeze_if_scalar(e)

    print('SpMM(message func: {}, reduce func: {})'.format(msg, reducer))
    v = dgl.sparse._gspmm(g, msg, reducer, u, e)[0]
    non_degree_indices = F.tensor(
        np.nonzero(F.asnumpy(g.in_degrees()) != 0)[0])
    v = F.gather_row(v, non_degree_indices)
    g.update_all(udf_msg[msg], udf_reduce[reducer])
    if 'v' in g.dstdata:
        v1 = F.gather_row(g.dstdata['v'], non_degree_indices)
        assert F.allclose(v, v1, rtol=1e-3, atol=1e-3)
    print('passed')

    g.srcdata.pop('x')
    g.edata.pop('w')
    if 'v' in g.dstdata: g.dstdata.pop('v')

@pytest.mark.parametrize('g', graphs)
@pytest.mark.parametrize('shp', sddmm_shapes)
@pytest.mark.parametrize('lhs_target', ['u', 'v', 'e'])
@pytest.mark.parametrize('rhs_target', ['u', 'v', 'e'])
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'dot', 'copy_lhs', 'copy_rhs'])
def test_sddmm(g, shp, lhs_target, rhs_target, msg):
    if dgl.backend.backend_name == 'mxnet' and g.number_of_edges() == 0:
        pytest.skip()   # mxnet do not support zero shape tensor
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
    lhs = F.tensor(np.random.rand(*lhs_shp) + 1)
    rhs = F.tensor(np.random.rand(*rhs_shp) + 1)
    print('lhs shape: {}, rhs shape: {}'.format(F.shape(lhs), F.shape(rhs)))

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
    lhs_frame['x'] = _unsqueeze_if_scalar(lhs)
    rhs_frame['y'] = _unsqueeze_if_scalar(rhs)
    msg_func = lhs_target + '_' + msg + '_' + rhs_target
    print('SDDMM(message func: {})'.format(msg_func))

    e = dgl.sparse._gsddmm(g, msg, lhs, rhs,
                           lhs_target=lhs_target, rhs_target=rhs_target)
    g.apply_edges(udf_apply_edges[msg_func])
    if 'm' in g.edata:
        e1 = g.edata['m']
        assert F.allclose(e, e1, rtol=1e-3, atol=1e-3)
    print('passed')

    lhs_frame.pop('x')
    rhs_frame.pop('y')
    if 'm' in g.edata: g.edata.pop('m')

