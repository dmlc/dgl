import pytest
import backend as F
import dgl
from dgl.base import is_internal_column

def check_fail(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return False
    except:
        return True

def assert_is_identical(g, g2):
    assert g.number_of_nodes() == g2.number_of_nodes()
    src, dst = g.all_edges(order='eid')
    src2, dst2 = g2.all_edges(order='eid')
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
        assert g.number_of_nodes(ntype) == g2.number_of_nodes(ntype)
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
        src, dst = g.all_edges(etype=etype, order='eid')
        src2, dst2 = g2.all_edges(etype=etype, order='eid')
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
