import backend as F

import dgl

__all__ = ["check_graph_equal"]


def check_graph_equal(g1, g2, *, check_idtype=True, check_feature=True):
    assert g1.device == g1.device
    if check_idtype:
        assert g1.idtype == g2.idtype
    assert g1.ntypes == g2.ntypes
    assert g1.etypes == g2.etypes
    assert g1.srctypes == g2.srctypes
    assert g1.dsttypes == g2.dsttypes
    assert g1.canonical_etypes == g2.canonical_etypes
    assert g1.batch_size == g2.batch_size

    # check if two metagraphs are identical
    for edges, features in g1.metagraph().edges(keys=True).items():
        assert g2.metagraph().edges(keys=True)[edges] == features

    for nty in g1.ntypes:
        assert g1.number_of_nodes(nty) == g2.number_of_nodes(nty)
        assert F.allclose(g1.batch_num_nodes(nty), g2.batch_num_nodes(nty))
    for ety in g1.canonical_etypes:
        assert g1.number_of_edges(ety) == g2.number_of_edges(ety)
        assert F.allclose(g1.batch_num_edges(ety), g2.batch_num_edges(ety))
        src1, dst1, eid1 = g1.edges(etype=ety, form="all")
        src2, dst2, eid2 = g2.edges(etype=ety, form="all")
        if check_idtype:
            assert F.allclose(src1, src2)
            assert F.allclose(dst1, dst2)
            assert F.allclose(eid1, eid2)
        else:
            assert F.allclose(src1, F.astype(src2, g1.idtype))
            assert F.allclose(dst1, F.astype(dst2, g1.idtype))
            assert F.allclose(eid1, F.astype(eid2, g1.idtype))

    if check_feature:
        for nty in g1.ntypes:
            if g1.number_of_nodes(nty) == 0:
                continue
            for feat_name in g1.nodes[nty].data.keys():
                assert F.allclose(
                    g1.nodes[nty].data[feat_name], g2.nodes[nty].data[feat_name]
                )
        for ety in g1.canonical_etypes:
            if g1.number_of_edges(ety) == 0:
                continue
            for feat_name in g2.edges[ety].data.keys():
                assert F.allclose(
                    g1.edges[ety].data[feat_name], g2.edges[ety].data[feat_name]
                )
