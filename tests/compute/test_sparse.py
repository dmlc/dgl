import dgl
import networkx as nx
import backend as F

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
    'dot': lambda edges: {'m': F.sum(edges.src['x'] * edges.dst['y'], 1, keepdims=True)},
    'copy_u': lambda edges: {'m': edges.src['x']},
}

udf_reduce = {
    'sum': lambda nodes: {'v': F.sum(nodes.mailbox['m'], 1)},
    'min': lambda nodes: {'v': F.min(nodes.mailbox['m'], 1)[0]},
    'max': lambda nodes: {'v': F.max(nodes.mailbox['m'], 1)[0]}
}

def test_spmm():
    nxg = nx.erdos_renyi_graph(100, 0.3)
    for i in range(nxg.number_of_nodes()):
        nxg.add_edge(i, i)
    g = dgl.graph(nxg)
    u = F.randn((g.number_of_nodes(), 1, 3, 1))
    e = F.randn((g.number_of_edges(), 4, 1, 3))
    g.ndata['x'] = u
    g.edata['w'] = e

    for msg in ['add', 'sub', 'mul', 'div', 'copy_u', 'copy_e']:
        for reducer in ['sum', 'min', 'max']:
            print('SpMM(message func: {}, reduce func: {})'.format(msg, reducer))
            v = dgl.gspmm(g, msg, reducer, u, e)[0]
            g.update_all(udf_msg[msg], udf_reduce[reducer])
            v1 = g.ndata['v']
            assert F.allclose(v, v1, rtol=1e-3, atol=1e-3)
            print('passed')

def test_sddmm():
    nxg = nx.erdos_renyi_graph(100, 0.3)
    for i in range(nxg.number_of_nodes()):
        nxg.add_edge(i, i)
    g = dgl.graph(nxg)
    u = F.randn((g.number_of_nodes(), 1, 3, 4))
    v = F.randn((g.number_of_nodes(), 3, 1, 4))
    g.ndata['x'] = u
    g.ndata['y'] = v

    for msg in ['add', 'sub', 'mul', 'div', 'dot', 'copy_u']:
        print('SDDMM(message func: {})'.format(msg))
        e = dgl.gsddmm(g, msg, u, v)
        g.apply_edges(udf_apply_edges[msg])
        e1 = g.edata['m']
        assert F.allclose(e, e1, rtol=1e-3, atol=1e-3)
        print('passed')

if __name__ == '__main__':
    test_spmm()
    test_sddmm()

