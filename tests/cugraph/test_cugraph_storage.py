import dgl
from dgl.contrib.cugraph.convert import cugraph_storage_from_heterograph
import torch as th
import cudf
import numpy as np


def create_dgl_graph():
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


def test_cugraphstore_basic_apis():
    from dgl.contrib.cugraph.cugraph_storage import CuGraphStorage

    gs = CuGraphStorage(num_nodes_dict={"drug": 3, "gene": 2, "disease": 1})
    # add node data
    drug_df = cudf.DataFrame(
        {"node_ids": [0, 1, 2], "node_feat": [0.1, 0.2, 0.3]}
    )
    gs.add_node_data(drug_df, "node_ids", ntype="drug")

    # add edges
    drug_interacts_drug_df = cudf.DataFrame(
        {"src": [0, 1], "dst": [1, 2], "edge_feat": [0.2, 0.4]}
    )
    drug_interacts_gene = cudf.DataFrame({"src": [0, 1], "dst": [0, 1]})
    drug_treats_disease = cudf.DataFrame({"src": [1], "dst": [0]})
    gs.add_edge_data(
        drug_interacts_drug_df,
        node_col_names=["src", "dst"],
        canonical_etype=("drug", "interacts", "drug"),
    )
    gs.add_edge_data(
        drug_interacts_gene,
        node_col_names=["src", "dst"],
        canonical_etype=("drug", "interacts", "gene"),
    )
    gs.add_edge_data(
        drug_treats_disease,
        node_col_names=["src", "dst"],
        canonical_etype=("drug", "treats", "disease"),
    )

    assert gs.num_nodes() == 6

    assert gs.num_edges(("drug", "interacts", "drug")) == 2
    assert gs.num_edges(("drug", "interacts", "gene")) == 2
    assert gs.num_edges(("drug", "treats", "disease")) == 1

    node_feat = (
        gs.get_node_storage(key="node_feat", ntype="drug")
        .fetch([0, 1, 2])
        .to("cpu")
        .numpy()
    )
    np.testing.assert_equal(node_feat, np.asarray([0.1, 0.2, 0.3]))

    edge_feat = (
        gs.get_edge_storage(
            key="edge_feat", etype=("drug", "interacts", "drug")
        )
        .fetch([0, 1])
        .to("cpu")
        .numpy()
    )
    np.testing.assert_equal(edge_feat, np.asarray([0.2, 0.4]))


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
        assert_same_sampling_len(g, cugraph_gs, nodes, fanout)
 

def assert_same_sampling_len(dgl_g, cugraph_gs, nodes, fanout, edge_dir):
    dgl_o = dgl_g.sample_neighbors(nodes, fanout=fanout, edge_dir=edge_dir)
    cugraph_o = cugraph_gs.sample_neighbors(
        nodes, fanout=fanout, edge_dir=edge_dir
    )
    assert cugraph_o.num_edges() == dgl_o.num_edges()
    for etype in dgl_o.canonical_etypes:
        assert dgl_o.num_edges(etype) == cugraph_o.num_edges(etype)
