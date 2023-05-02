import io
import multiprocessing as mp
import os
import pickle
import unittest

import backend as F

import dgl
import dgl.function as fn
import networkx as nx
import scipy.sparse as ssp
from dgl.graph_index import create_graph_index
from dgl.utils import toindex
from utils import parametrize_idtype


def create_test_graph(idtype):
    g = dgl.heterograph(
        (
            {
                ("user", "follows", "user"): ([0, 1], [1, 2]),
                ("user", "plays", "game"): ([0, 1, 2, 1], [0, 0, 1, 1]),
                ("user", "wishes", "game"): ([0, 2], [1, 0]),
                ("developer", "develops", "game"): ([0, 1], [0, 1]),
            }
        ),
        idtype=idtype,
    )
    return g


def _assert_is_identical_hetero(g, g2):
    assert g.ntypes == g2.ntypes
    assert g.canonical_etypes == g2.canonical_etypes

    # check if two metagraphs are identical
    for edges, features in g.metagraph().edges(keys=True).items():
        assert g2.metagraph().edges(keys=True)[edges] == features

    # check if node ID spaces and feature spaces are equal
    for ntype in g.ntypes:
        assert g.num_nodes(ntype) == g2.num_nodes(ntype)

    # check if edge ID spaces and feature spaces are equal
    for etype in g.canonical_etypes:
        src, dst = g.all_edges(etype=etype, order="eid")
        src2, dst2 = g2.all_edges(etype=etype, order="eid")
        assert F.array_equal(src, src2)
        assert F.array_equal(dst, dst2)


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
@parametrize_idtype
def test_single_process(idtype):
    hg = create_test_graph(idtype=idtype)
    hg_share = hg.shared_memory("hg")
    hg_rebuild = dgl.hetero_from_shared_memory("hg")
    hg_save_again = hg_rebuild.shared_memory("hg")
    _assert_is_identical_hetero(hg, hg_share)
    _assert_is_identical_hetero(hg, hg_rebuild)
    _assert_is_identical_hetero(hg, hg_save_again)


def sub_proc(hg_origin, name):
    hg_rebuild = dgl.hetero_from_shared_memory(name)
    hg_save_again = hg_rebuild.shared_memory(name)
    _assert_is_identical_hetero(hg_origin, hg_rebuild)
    _assert_is_identical_hetero(hg_origin, hg_save_again)


@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
@parametrize_idtype
def test_multi_process(idtype):
    hg = create_test_graph(idtype=idtype)
    hg_share = hg.shared_memory("hg1")
    p = mp.Process(target=sub_proc, args=(hg, "hg1"))
    p.start()
    p.join()


@unittest.skipIf(
    F._default_context_str == "cpu", reason="Need gpu for this test"
)
@unittest.skipIf(
    dgl.backend.backend_name == "tensorflow",
    reason="Not support tensorflow for now",
)
def test_copy_from_gpu():
    hg = create_test_graph(idtype=F.int32)
    hg_gpu = hg.to(F.cuda())
    hg_share = hg_gpu.shared_memory("hg_gpu")
    p = mp.Process(target=sub_proc, args=(hg, "hg_gpu"))
    p.start()
    p.join()


# TODO: Test calling shared_memory with Blocks (a subclass of HeteroGraph)
if __name__ == "__main__":
    test_single_process(F.int64)
    test_multi_process(F.int32)
    test_copy_from_gpu()
