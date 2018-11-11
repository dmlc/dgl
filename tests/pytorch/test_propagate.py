import dgl
import networkx as nx
import torch as th
import utils as U

def mfunc(edges):
    return {'m' : edges.src['x']}

def rfunc(nodes):
    msg = th.sum(nodes.mailbox['m'], 1)
    return {'x' : nodes.data['x'] + msg}

def test_prop_nodes_bfs():
    g = dgl.DGLGraph(nx.path_graph(5))
    g.ndata['x'] = th.ones((5, 2))
    g.register_message_func(mfunc)
    g.register_reduce_func(rfunc)

    dgl.prop_nodes_bfs(g, 0)
    # pull nodes using bfs order will result in a cumsum[i] + data[i] + data[i+1]
    assert U.allclose(g.ndata['x'],
            th.tensor([[2., 2.], [4., 4.], [6., 6.], [8., 8.], [9., 9.]]))

def test_prop_edges_dfs():
    g = dgl.DGLGraph(nx.path_graph(5))
    g.register_message_func(mfunc)
    g.register_reduce_func(rfunc)

    g.ndata['x'] = th.ones((5, 2))
    dgl.prop_edges_dfs(g, 0)
    # snr using dfs results in a cumsum
    assert U.allclose(g.ndata['x'],
            th.tensor([[1., 1.], [2., 2.], [3., 3.], [4., 4.], [5., 5.]]))

    g.ndata['x'] = th.ones((5, 2))
    dgl.prop_edges_dfs(g, 0, has_reverse_edge=True)
    # result is cumsum[i] + cumsum[i-1]
    assert U.allclose(g.ndata['x'],
            th.tensor([[1., 1.], [3., 3.], [5., 5.], [7., 7.], [9., 9.]]))

    g.ndata['x'] = th.ones((5, 2))
    dgl.prop_edges_dfs(g, 0, has_nontree_edge=True)
    # result is cumsum[i] + cumsum[i+1]
    assert U.allclose(g.ndata['x'],
            th.tensor([[3., 3.], [5., 5.], [7., 7.], [9., 9.], [5., 5.]]))

def test_prop_nodes_topo():
    # bi-directional chain
    g = dgl.DGLGraph(nx.path_graph(5))
    assert U.check_fail(dgl.prop_nodes_topo, g)  # has loop

    # tree
    tree = dgl.DGLGraph()
    tree.add_nodes(5)
    tree.add_edge(1, 0)
    tree.add_edge(2, 0)
    tree.add_edge(3, 2)
    tree.add_edge(4, 2)
    tree.register_message_func(mfunc)
    tree.register_reduce_func(rfunc)
    # init node feature data
    tree.ndata['x'] = th.zeros((5, 2))
    # set all leaf nodes to be ones
    tree.nodes[[1, 3, 4]].data['x'] = th.ones((3, 2))
    dgl.prop_nodes_topo(tree)
    # root node get the sum
    assert U.allclose(tree.nodes[0].data['x'], th.tensor([[3., 3.]]))

if __name__ == '__main__':
    test_prop_nodes_bfs()
    test_prop_edges_dfs()
    test_prop_nodes_topo()
