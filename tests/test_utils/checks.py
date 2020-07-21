import dgl
import backend as F

__all__ = ['check_graph_equal']

def check_graph_equal(g1, g2, check_feature=True):
    assert g1.device == g1.device
    assert g1.idtype == g2.idtype
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
        src1, dst1, eid1 = g1.edges(etype=ety, form='all')
        src2, dst2, eid2 = g2.edges(etype=ety, form='all')
        assert F.allclose(src1, src2)
        assert F.allclose(dst1, dst2)
        assert F.allclose(eid1, eid2)

    if check_feature:
        for nty in g1.ntypes:
            if g1.number_of_nodes(nty) == 0:
                continue
            for feat_name in g1.nodes[nty].data.keys():
                assert F.allclose(g1.nodes[nty].data[feat_name], g2.nodes[nty].data[feat_name])
        for ety in g1.canonical_etypes:
            if g1.number_of_edges(ety) == 0:
                continue
            for feat_name in g2.edges[ety].data.keys():
                assert F.allclose(g1.edges[ety].data[feat_name], g2.edges[ety].data[feat_name])
