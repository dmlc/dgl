# NOTE(vibwu): Currently cugraph must be imported before torch to avoid a resource cleanup issue.
#    See https://github.com/rapidsai/cugraph/issues/2718
import cugraph  # usort: skip
import backend as F

import dgl


def test_dummy():
    cg = cugraph.Graph()
    assert cg is not None


def test_to_cugraph_conversion():
    g = dgl.graph((F.tensor([0, 1, 2, 3]), F.tensor([1, 0, 3, 2]))).to("cuda")
    cugraph_g = g.to_cugraph()

    assert cugraph_g.number_of_nodes() == g.num_nodes()
    assert cugraph_g.number_of_edges() == g.num_edges()

    assert cugraph_g.has_edge(0, 1)
    assert cugraph_g.has_edge(1, 0)
    assert cugraph_g.has_edge(3, 2)


def test_from_cugraph_conversion():
    # cudf is a dependency of cugraph
    import cudf

    # directed graph conversion test
    cugraph_g = cugraph.Graph(directed=True)
    df = cudf.DataFrame({"source": [0, 1, 2, 3], "destination": [1, 2, 3, 2]})

    cugraph_g.from_cudf_edgelist(df)

    g = dgl.from_cugraph(cugraph_g)

    assert g.device.type == "cuda"
    assert g.num_nodes() == cugraph_g.number_of_nodes()
    assert g.num_edges() == cugraph_g.number_of_edges()

    # assert reverse edges are not present
    assert g.has_edges_between(0, 1)
    assert not g.has_edges_between(1, 0)
    assert g.has_edges_between(1, 2)
    assert not g.has_edges_between(2, 1)
    assert g.has_edges_between(2, 3)

    # undirected graph conversion test
    cugraph_g = cugraph.Graph(directed=False)
    df = cudf.DataFrame({"source": [0, 1, 2, 3], "destination": [1, 2, 3, 2]})

    cugraph_g.from_cudf_edgelist(df)

    g = dgl.from_cugraph(cugraph_g)

    assert g.device.type == "cuda"
    assert g.num_nodes() == cugraph_g.number_of_nodes()
    # assert reverse edges are present
    assert g.has_edges_between(0, 1)
    assert g.has_edges_between(1, 0)
    assert g.has_edges_between(1, 2)
    assert g.has_edges_between(2, 1)
    assert g.has_edges_between(2, 3)
