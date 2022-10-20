import numpy as np
import dgl
from dgl.contrib.cugraph.convert import cugraph_storage_from_heterograph


def create_dgl_graph():
    import dgl
    import torch as th

    graph_data = {
        ("nt.a", "connects", "nt.b"): (
            th.tensor([0, 1, 2]),
            th.tensor([0, 1, 2]),
        ),
        ("nt.a", "connects", "nt.c"): (
            th.tensor([0, 1, 2]),
            th.tensor([0, 1, 2]),
        ),
        ("nt.c", "connects", "nt.c"): (
            th.tensor([1, 3, 4, 5]),
            th.tensor([0, 0, 0, 0]),
        ),
    }
    g = dgl.heterograph(graph_data)
    return g


def assert_same_sampling_len(dgl_g, cugraph_gs, nodes, fanout, edge_dir):
    dgl_o = dgl_g.sample_neighbors(nodes, fanout=fanout, edge_dir=edge_dir)
    cugraph_o = cugraph_gs.sample_neighbors(
        nodes, fanout=fanout, edge_dir=edge_dir
    )
    assert cugraph_o.num_edges() == dgl_o.num_edges()
    for etype in dgl_o.canonical_etypes:
        assert dgl_o.num_edges(etype) == cugraph_o.num_edges(etype)


def test_sampling_heterograph():
    dgl_g = create_dgl_graph()
    cugraph_gs = cugraph_storage_from_heterograph(dgl_g)

    for fanout in [1, 2, 3, -1]:
        for ntype in ["nt.a", "nt.b", "nt.c"]:
            for d in ["in", "out"]:
                assert_same_sampling_len(
                    dgl_g,
                    cugraph_gs,
                    nodes={ntype: [0]},
                    fanout=fanout,
                    edge_dir=d,
                )


def test_sampling_homogenous():
    src_ar = np.asarray([0, 1, 2, 0, 1, 2, 7, 9, 10, 11], dtype=np.int32)
    dst_ar = np.asarray([3, 4, 5, 6, 7, 8, 6, 6, 6, 6], dtype=np.int32)
    g = dgl.heterograph({("a", "connects", "a"): (src_ar, dst_ar)})
    cugraph_gs = cugraph_storage_from_heterograph(g)
    # Convert to homogeneous
    g = dgl.to_homogeneous(g)
    nodes = [6]
    # Test for multiple fanouts
    for fanout in [1, 2, 3]:
        exp_g = g.sample_neighbors(nodes, fanout=fanout)
        cu_g = cugraph_gs.sample_neighbors(nodes, fanout=fanout)
        exp_src, exp_dst = exp_g.edges()
        cu_src, cu_dst = cu_g.edges()
        assert len(exp_src) == len(cu_src)

    # Test same results for all neighbours
    exp_g = g.sample_neighbors(nodes, fanout=-1)
    cu_g = cugraph_gs.sample_neighbors(nodes, fanout=-1)
    exp_src, exp_dst = exp_g.edges()
    exp_src, exp_dst = exp_src.numpy(), exp_dst.numpy()

    cu_src, cu_dst = cu_g.edges()
    cu_src, cu_dst = cu_src.to("cpu").numpy(), cu_dst.to("cpu").numpy()

    np.testing.assert_equal(exp_src, cu_src)
    np.testing.assert_equal(exp_dst, cu_dst)
