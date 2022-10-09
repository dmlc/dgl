from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage
from cugraph.experimental import PropertyGraph
import cudf
import cupy as cp
import numpy as np


def create_gs_heterogeneous_dgl_eg():
    pg = PropertyGraph()
    gs = CuGraphStorage(pg)

    # Add Edge Data
    src_ser = [0, 1, 2, 0, 1, 2, 7, 9, 10, 11]
    dst_ser = [3, 4, 5, 6, 7, 8, 6, 6, 6, 6]
    etype_ser = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
    edge_feat = [10, 10, 10, 11, 11, 11, 12, 12, 12, 13]

    etype_map = {
        0: ("nt.a", "connects", "nt.b"),
        1: ("nt.a", "connects", "nt.c"),
        2: ("nt.c", "connects", "nt.c"),
    }

    df = cudf.DataFrame(
        {
            "src": src_ser,
            "dst": dst_ser,
            "etype": etype_ser,
            "edge_feat": edge_feat,
        }
    )
    df = df.astype(np.int64)
    for e in df["etype"].unique().values_host:
        subset_df = df[df["etype"] == e][
            ["src", "dst", "edge_feat"]
        ].reset_index(drop=True)
        gs.add_edge_data(
            subset_df,
            ["src", "dst"],
            feat_name="edge_feat",
            canonical_etype=etype_map[e],
        )

    node_ser = np.arange(0, 12)
    node_type = ["nt.a"] * 3 + ["nt.b"] * 3 + ["nt.c"] * 6
    node_feat = np.arange(0, 12) * 10

    df = cudf.DataFrame(
        {"node_id": node_ser, "ntype": node_type, "node_feat": node_feat}
    )

    for n in df["ntype"].unique().values_host:
        subset_df = df[df["ntype"] == n][["node_id", "node_feat"]]
        gs.add_node_data(subset_df, "node_id", feat_name="node_feat", ntype=n)

    return gs


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


def test_sampling_len_cugraph():
    dgl_g = create_dgl_graph()
    cugraph_gs = create_gs_heterogeneous_dgl_eg()
    # vertex_d = cugraph_gs.graphstore.gdata.renumber_vertices_by_type()
    # edge_d = cugraph_gs.graphstore.gdata.renumber_edges_by_type()

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
