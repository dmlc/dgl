import dgl
import pytest
import networkx as nx
import backend as F
import numpy as np 

np.random.seed(42)
dgl.random.seed(42)

def _unsqueeze_if_scalar(x):    # used in udf, to unsqueeze the feature if it's scalar
    return x if F.ndim(x) > 1 else F.unsqueeze(x, -1)

def _rand_operand_1(shp):
    return F.tensor(np.random.rand(*shp))

def _rand_operand_2(shp):   # for division op, the divisor should be greater than 1
    return F.tensor(np.random.rand(*shp) + 1)

udf_msg = {
    'add': lambda edges: {'m': edges.src['x'] + edges.data['w']},
    'sub': lambda edges: {'m': edges.src['x'] - edges.data['w']},
    'mul': lambda edges: {'m': edges.src['x'] * edges.data['w']},
    'div': lambda edges: {'m': edges.src['x'] / edges.data['w']},
    'copy_u': lambda edges: {'m': edges.src['x']},
    'copy_e': lambda edges: {'m': edges.data['w']}
}

udf_apply_edges = {
    'add': lambda edges: {'m': edges.src['x'] + edges.dst['y']},
    'sub': lambda edges: {'m': edges.src['x'] - edges.dst['y']},
    'mul': lambda edges: {'m': edges.src['x'] * edges.dst['y']},
    'div': lambda edges: {'m': edges.src['x'] / edges.dst['y']},
    'dot': lambda edges: {'m': F.sum(edges.src['x'] * edges.dst['y'], -1, keepdims=True)},
    'copy_u': lambda edges: {'m': edges.src['x']},
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
    u = _rand_operand_1((g.number_of_src_nodes(),) + shp[0])
    e = _rand_operand_2((g.number_of_edges(),) + shp[1])
    print('u shape: {}, e shape: {}'.format(F.shape(u), F.shape(e)))
    g.srcdata['x'] = _unsqueeze_if_scalar(u)
    g.edata['w'] = _unsqueeze_if_scalar(e)

    print('SpMM(message func: {}, reduce func: {})'.format(msg, reducer))
    v = dgl.gspmm(g, msg, reducer, u, e)[0]
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
@pytest.mark.parametrize('msg', ['add', 'sub', 'mul', 'div', 'dot', 'copy_u'])
def test_sddmm(g, shp, msg):
    if dgl.backend.backend_name == 'mxnet' and g.number_of_edges() == 0:
        pytest.skip()   # mxnet do not support zero shape tensor
    print(g)
    u = _rand_operand_1((g.number_of_src_nodes(),) + shp[0])
    v = _rand_operand_2((g.number_of_dst_nodes(),) + shp[1])
    print('u shape: {}, v shape: {}'.format(F.shape(u), F.shape(v)))
    g.srcdata['x'] = _unsqueeze_if_scalar(u)
    g.dstdata['y'] = _unsqueeze_if_scalar(v)

    print('SDDMM(message func: {})'.format(msg))
    e = dgl.gsddmm(g, msg, u, v)
    g.apply_edges(udf_apply_edges[msg])
    if 'm' in g.edata:
        e1 = g.edata['m']
        assert F.allclose(e, e1, rtol=1e-3, atol=1e-3)
    print('passed')

    g.srcdata.pop('x')
    g.dstdata.pop('y')
    if 'm' in g.edata: g.edata.pop('m')

