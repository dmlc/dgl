import dgl
import backend as F

from dgl.base import ALL

def check_equivalence_between_heterographs(g1, g2, node_attrs=None, edge_attrs=None):
    assert g1.ntypes == g2.ntypes
    assert g1.etypes == g2.etypes
    assert g1.canonical_etypes == g2.canonical_etypes

    for nty in g1.ntypes:
        assert g1.number_of_nodes(nty) == g2.number_of_nodes(nty)

    for ety in g1.etypes:
        if len(g1._etype2canonical[ety]) > 0:
            assert g1.number_of_edges(ety) == g2.number_of_edges(ety)

    for ety in g1.canonical_etypes:
        assert g1.number_of_edges(ety) == g2.number_of_edges(ety)
        src1, dst1 = g1.all_edges(etype=ety)
        src2, dst2 = g2.all_edges(etype=ety)
        assert F.allclose(src1, src2)
        assert F.allclose(dst1, dst2)

    if node_attrs is not None:
        for nty in node_attrs.keys():
            for feat_name in node_attrs[nty]:
                assert F.allclose(g1.nodes[nty].data[feat_name], g2.nodes[nty].data[feat_name])

    if edge_attrs is not None:
        for ety in edge_attrs.keys():
            for feat_name in edge_attrs[ety]:
                assert F.allclose(g1.edges[ety].data[feat_name], g2.edges[ety].data[feat_name])

def test_batching_hetero_topology():
    """Test batching two DGLHeteroGraphs where some nodes are isolated in some relations"""
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'follows', 'developer'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1), (3, 1)]
    })
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'follows', 'developer'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0), (2, 1)]
    })
    bg = dgl.batch_hetero([g1, g2])

    assert bg.ntypes == g2.ntypes
    assert bg.etypes == g2.etypes
    assert bg.canonical_etypes == g2.canonical_etypes
    assert bg.batch_size == 2

    # Test number of nodes
    for ntype in bg.ntypes:
        assert bg.batch_num_nodes(ntype) == [
            g1.number_of_nodes(ntype), g2.number_of_nodes(ntype)]
        assert bg.number_of_nodes(ntype) == (
                g1.number_of_nodes(ntype) + g2.number_of_nodes(ntype))

    # Test number of edges
    assert bg.batch_num_edges('plays') == [
        g1.number_of_edges('plays'), g2.number_of_edges('plays')]
    assert bg.number_of_edges('plays') == (
        g1.number_of_edges('plays') + g2.number_of_edges('plays'))

    for etype in bg.canonical_etypes:
        assert bg.batch_num_edges(etype) == [
            g1.number_of_edges(etype), g2.number_of_edges(etype)]
        assert bg.number_of_edges(etype) == (
            g1.number_of_edges(etype) + g2.number_of_edges(etype))

    # Test relabeled nodes
    for ntype in bg.ntypes:
        assert list(F.asnumpy(bg.nodes(ntype))) == list(range(bg.number_of_nodes(ntype)))

    # Test relabeled edges
    src, dst = bg.all_edges(etype=('user', 'follows', 'user'))
    assert list(F.asnumpy(src)) == [0, 1, 4, 5]
    assert list(F.asnumpy(dst)) == [1, 2, 5, 6]
    src, dst = bg.all_edges(etype=('user', 'follows', 'developer'))
    assert list(F.asnumpy(src)) == [0, 1, 4, 5]
    assert list(F.asnumpy(dst)) == [1, 2, 4, 5]
    src, dst = bg.all_edges(etype='plays')
    assert list(F.asnumpy(src)) == [0, 1, 2, 3, 4, 5, 6]
    assert list(F.asnumpy(dst)) == [0, 0, 1, 1, 2, 2, 3]

    # Test unbatching graphs
    g3, g4 = dgl.unbatch_hetero(bg)
    check_equivalence_between_heterographs(g1, g3)
    check_equivalence_between_heterographs(g2, g4)

def test_batching_hetero_and_batched_hetero_topology():
    """Test batching a DGLHeteroGraph and a BatchedDGLHeteroGraph."""
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    })
    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    })
    bg1 = dgl.batch_hetero([g1, g2])
    g3 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1)],
        ('user', 'plays', 'game'): [(1, 0)]
    })
    bg2 = dgl.batch_hetero([bg1, g3])
    assert bg2.ntypes == g3.ntypes
    assert bg2.etypes == g3.etypes
    assert bg2.canonical_etypes == g3.canonical_etypes
    assert bg2.batch_size == 3

    # Test number of nodes
    for ntype in bg2.ntypes:
        assert bg2.batch_num_nodes(ntype) == [
            g1.number_of_nodes(ntype), g2.number_of_nodes(ntype), g3.number_of_nodes(ntype)]
        assert bg2.number_of_nodes(ntype) == (
                g1.number_of_nodes(ntype) + g2.number_of_nodes(ntype) + g3.number_of_nodes(ntype))

    # Test number of edges
    for etype in bg2.etypes:
        assert bg2.batch_num_edges(etype) == [
            g1.number_of_edges(etype), g2.number_of_edges(etype), g3.number_of_edges(etype)]
        assert bg2.number_of_edges(etype) == (
                g1.number_of_edges(etype) + g2.number_of_edges(etype) + g3.number_of_edges(etype))

    for etype in bg2.canonical_etypes:
        assert bg2.batch_num_edges(etype) == [
            g1.number_of_edges(etype), g2.number_of_edges(etype), g3.number_of_edges(etype)]
        assert bg2.number_of_edges(etype) == (
                g1.number_of_edges(etype) + g2.number_of_edges(etype) + g3.number_of_edges(etype))

    # Test relabeled nodes
    for ntype in bg2.ntypes:
        assert list(F.asnumpy(bg2.nodes(ntype))) == list(range(bg2.number_of_nodes(ntype)))

    # Test relabeled edges
    src, dst = bg2.all_edges(etype='follows')
    assert list(F.asnumpy(src)) == [0, 1, 3, 4, 6]
    assert list(F.asnumpy(dst)) == [1, 2, 4, 5, 7]
    src, dst = bg2.all_edges(etype='plays')
    assert list(F.asnumpy(src)) == [0, 1, 3, 4, 7]
    assert list(F.asnumpy(dst)) == [0, 0, 1, 1, 2]

    # Test unbatching graphs
    g4, g5, g6 = dgl.unbatch_hetero(bg2)
    check_equivalence_between_heterographs(g1, g4)
    check_equivalence_between_heterographs(g2, g5)
    check_equivalence_between_heterographs(g3, g6)

def test_batched_features():
    """Test the features of batched DGLHeteroGraphs"""
    g1 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    })
    g1.nodes['user'].data['h1'] = F.tensor([[0.], [1.], [2.]])
    g1.nodes['user'].data['h2'] = F.tensor([[3.], [4.], [5.]])
    g1.nodes['game'].data['h1'] = F.tensor([[0.]])
    g1.nodes['game'].data['h2'] = F.tensor([[1.]])
    g1.edges['follows'].data['h1'] = F.tensor([[0.], [1.]])
    g1.edges['follows'].data['h2'] = F.tensor([[2.], [3.]])
    g1.edges['plays'].data['h1'] = F.tensor([[0.], [1.]])

    g2 = dgl.heterograph({
        ('user', 'follows', 'user'): [(0, 1), (1, 2)],
        ('user', 'plays', 'game'): [(0, 0), (1, 0)]
    })
    g2.nodes['user'].data['h1'] = F.tensor([[0.], [1.], [2.]])
    g2.nodes['user'].data['h2'] = F.tensor([[3.], [4.], [5.]])
    g2.nodes['game'].data['h1'] = F.tensor([[0.]])
    g2.nodes['game'].data['h2'] = F.tensor([[1.]])
    g2.edges['follows'].data['h1'] = F.tensor([[0.], [1.]])
    g2.edges['follows'].data['h2'] = F.tensor([[2.], [3.]])
    g2.edges['plays'].data['h1'] = F.tensor([[0.], [1.]])

    bg = dgl.batch_hetero([g1, g2],
                          node_attrs=ALL,
                          edge_attrs={
                              ('user', 'follows', 'user'): 'h1',
                              ('user', 'plays', 'game'): None
                          })

    assert F.allclose(bg.nodes['user'].data['h1'],
                      F.cat([g1.nodes['user'].data['h1'], g2.nodes['user'].data['h1']], dim=0))
    assert F.allclose(bg.nodes['user'].data['h2'],
                      F.cat([g1.nodes['user'].data['h2'], g2.nodes['user'].data['h2']], dim=0))
    assert F.allclose(bg.nodes['game'].data['h1'],
                      F.cat([g1.nodes['game'].data['h1'], g2.nodes['game'].data['h1']], dim=0))
    assert F.allclose(bg.nodes['game'].data['h2'],
                      F.cat([g1.nodes['game'].data['h2'], g2.nodes['game'].data['h2']], dim=0))
    assert F.allclose(bg.edges['follows'].data['h1'],
                      F.cat([g1.edges['follows'].data['h1'], g2.edges['follows'].data['h1']], dim=0))
    assert 'h2' not in bg.edges['follows'].data.keys()
    assert 'h1' not in bg.edges['plays'].data.keys()

    # Test unbatching graphs
    g3, g4 = dgl.unbatch_hetero(bg)
    check_equivalence_between_heterographs(
        g1, g3,
        node_attrs={'user': ['h1', 'h2'], 'game': ['h1', 'h2']},
        edge_attrs={('user', 'follows', 'user'): ['h1']})
    check_equivalence_between_heterographs(
        g2, g4,
        node_attrs={'user': ['h1', 'h2'], 'game': ['h1', 'h2']},
        edge_attrs={('user', 'follows', 'user'): ['h1']})

if __name__ == '__main__':
    test_batching_hetero_topology()
    test_batching_hetero_and_batched_hetero_topology()
    test_batched_features()
