import dgl
import dgl.function as fn
import torch as th
import utils as U

def test_graph_structure():
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 1, 2], [1, 2, 1])
    rg = dgl.reverse(g)

    assert g._readonly == rg._readonly
    assert g.is_multigraph == rg.is_multigraph

    assert g.number_of_nodes() == rg.number_of_nodes()
    assert len(g) == len(rg)
    assert U.allclose(rg.has_nodes(g.nodes()).float(), th.ones(3))

    assert g.number_of_edges() == rg.number_of_edges()
    assert U.allclose(rg.has_edges_between([1, 2, 1], [0, 1, 2]).float(), th.ones(3))
    assert g.edge_id(0, 1) == rg.edge_id(1, 0)
    assert g.edge_id(1, 2) == rg.edge_id(2, 1)
    assert g.edge_id(2, 1) == rg.edge_id(1, 2)

def test_shared_frames():
    g = dgl.DGLGraph()
    g.add_nodes(3)
    g.add_edges([0, 1, 2], [1, 2, 1])
    g.ndata['h'] = th.tensor([[0.], [1.], [2.]], requires_grad=True)
    g.edata['h'] = th.tensor([[3.], [4.], [5.]], requires_grad=True)

    rg = dgl.reverse(g)
    assert U.allclose(g.ndata['h'], rg.ndata['h'])
    assert U.allclose(g.edata['h'], rg.edata['h'])
    assert U.allclose(g.edges[[0, 2], [1, 1]].data['h'],
                      rg.edges[[1, 1], [0, 2]].data['h'])

    rg.ndata['h'] = rg.ndata['h'] + 1
    assert U.allclose(rg.ndata['h'], g.ndata['h'])

    g.edata['h'] = g.edata['h'] - 1
    assert U.allclose(rg.edata['h'], g.edata['h'])

    src_msg = fn.copy_src(src='h', out='m')
    sum_reduce = fn.sum(msg='m', out='h')

    rg.update_all(src_msg, sum_reduce)
    assert U.allclose(g.ndata['h'], rg.ndata['h'])

    # Grad check
    g.ndata['h'].retain_grad()
    rg.ndata['h'].retain_grad()
    loss_func = th.nn.MSELoss()
    target = th.zeros(3, 1)
    loss = loss_func(rg.ndata['h'], target)
    loss.backward()
    assert U.allclose(g.ndata['h'].grad, rg.ndata['h'].grad)

    # Dynamically modify the original graph and the reverse.
    g.add_nodes(2)
    assert g.number_of_nodes() == rg.number_of_nodes()
    assert U.allclose(g.ndata['h'], rg.ndata['h'])
    rg.add_nodes(1)
    assert g.number_of_nodes() == rg.number_of_nodes()
    assert U.allclose(g.ndata['h'], rg.ndata['h'])

    g.add_edges([3, 4], [4, 5])
    g.edges[[3, 4], [4, 5]].data['h'] = th.tensor([[6.], [7.]])
    assert g.number_of_edges() == rg.number_of_edges()
    assert U.allclose(rg.has_edges_between([4, 5], [3, 4]).float(), th.ones(2))
    assert U.allclose(g.edata['h'], rg.edata['h'])
    rg.add_edge(0, 5)
    rg.edges[[0], [5]].data['h'] = th.tensor([[8.]])
    assert g.number_of_edges() == rg.number_of_edges()
    assert g.has_edges_between(5, 0)
    assert U.allclose(g.edata['h'], rg.edata['h'])

if __name__ == '__main__':
    test_graph_structure()
    test_shared_frames()
