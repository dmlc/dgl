import backend as F

import dgl
import pytest
from dgl.base import is_internal_column

__all__ = [
    "check_fail",
    "assert_is_identical",
    "assert_is_identical_hetero",
    "check_graph_equal",
]


def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True


def assert_is_identical(g, g2):
    assert g.num_nodes() == g2.num_nodes()
    src, dst = g.all_edges(order="eid")
    src2, dst2 = g2.all_edges(order="eid")
    assert F.array_equal(src, src2)
    assert F.array_equal(dst, dst2)

    assert len(g.ndata) == len(g2.ndata)
    assert len(g.edata) == len(g2.edata)
    for k in g.ndata:
        assert F.allclose(g.ndata[k], g2.ndata[k])
    for k in g.edata:
        assert F.allclose(g.edata[k], g2.edata[k])


def assert_is_identical_hetero(g, g2, ignore_internal_data=False):
    assert g.ntypes == g2.ntypes
    assert g.canonical_etypes == g2.canonical_etypes

    # check if two metagraphs are identical
    for edges, features in g.metagraph().edges(keys=True).items():
        assert g2.metagraph().edges(keys=True)[edges] == features

    # check if node ID spaces and feature spaces are equal
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == g2.num_nodes(ntype)
        if ignore_internal_data:
            for k in list(g.nodes[ntype].data.keys()):
                if is_internal_column(k):
                    del g.nodes[ntype].data[k]
            for k in list(g2.nodes[ntype].data.keys()):
                if is_internal_column(k):
                    del g2.nodes[ntype].data[k]
        assert len(g.nodes[ntype].data) == len(g2.nodes[ntype].data)
        for k in g.nodes[ntype].data:
            assert F.allclose(g.nodes[ntype].data[k], g2.nodes[ntype].data[k])

    # check if edge ID spaces and feature spaces are equal
    for etype in g.canonical_etypes:
        src, dst = g.all_edges(etype=etype, order="eid")
        src2, dst2 = g2.all_edges(etype=etype, order="eid")
        assert F.array_equal(src, src2)
        assert F.array_equal(dst, dst2)
        if ignore_internal_data:
            for k in list(g.edges[etype].data.keys()):
                if is_internal_column(k):
                    del g.edges[etype].data[k]
            for k in list(g2.edges[etype].data.keys()):
                if is_internal_column(k):
                    del g2.edges[etype].data[k]
        assert len(g.edges[etype].data) == len(g2.edges[etype].data)
        for k in g.edges[etype].data:
            assert F.allclose(g.edges[etype].data[k], g2.edges[etype].data[k])


def check_graph_equal(g1, g2, *, check_idtype=True, check_feature=True):
    assert g1.device == g2.device
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
        assert g1.num_nodes(nty) == g2.num_nodes(nty)
        assert F.allclose(g1.batch_num_nodes(nty), g2.batch_num_nodes(nty))
    for ety in g1.canonical_etypes:
        assert g1.num_edges(ety) == g2.num_edges(ety)
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
            if g1.num_nodes(nty) == 0:
                continue
            for feat_name in g1.nodes[nty].data.keys():
                assert F.allclose(
                    g1.nodes[nty].data[feat_name], g2.nodes[nty].data[feat_name]
                )
        for ety in g1.canonical_etypes:
            if g1.num_edges(ety) == 0:
                continue
            for feat_name in g2.edges[ety].data.keys():
                assert F.allclose(
                    g1.edges[ety].data[feat_name], g2.edges[ety].data[feat_name]
                )
