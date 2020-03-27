import backend as F
import numpy as np
from scipy import sparse as spsp
from dgl import DGLError
from dgl.utils import toindex
from dgl.graph_index import create_graph_index
from dgl.graph_index import from_scipy_sparse_matrix

def test_node_subgraph():
    gi = create_graph_index(None, False)
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(0, 2)
    gi.add_edge(0, 3)

    sub2par_nodemap = [2, 0, 3]
    sgi = gi.node_subgraph(toindex(sub2par_nodemap))

    for s, d, e in zip(*sgi.graph.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])

def test_edge_subgraph():
    gi = create_graph_index(None, False)
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 3)

    sub2par_edgemap = [3, 2]
    sgi = gi.edge_subgraph(toindex(sub2par_edgemap))

    for s, d, e in zip(*sgi.graph.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])

def test_edge_subgraph_preserve_nodes():
    gi = create_graph_index(None, False)
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 3)

    sub2par_edgemap = [3, 2]
    sgi = gi.edge_subgraph(toindex(sub2par_edgemap), preserve_nodes=True)

    assert len(sgi.induced_nodes.tonumpy()) == 4

    for s, d, e in zip(*sgi.graph.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])


def test_immutable_edge_subgraph():
    gi = create_graph_index(None, False)
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 3)
    gi.readonly() # Make the graph readonly

    sub2par_edgemap = [3, 2]
    sgi = gi.edge_subgraph(toindex(sub2par_edgemap))

    for s, d, e in zip(*sgi.graph.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])

def test_immutable_edge_subgraph_preserve_nodes():
    gi = create_graph_index(None, False)
    gi.add_nodes(4)
    gi.add_edge(0, 1)
    gi.add_edge(0, 1)
    gi.add_edge(0, 2)
    gi.add_edge(2, 3)
    gi.readonly()

    sub2par_edgemap = [3, 2]
    sgi = gi.edge_subgraph(toindex(sub2par_edgemap), preserve_nodes=True)

    assert len(sgi.induced_nodes.tonumpy()) == 4

    for s, d, e in zip(*sgi.graph.edges()):
        assert sgi.induced_edges[e] in gi.edge_id(
                sgi.induced_nodes[s], sgi.induced_nodes[d])

def create_large_graph_index(num_nodes):
    row = np.random.choice(num_nodes, num_nodes * 10)
    col = np.random.choice(num_nodes, num_nodes * 10)
    spm = spsp.coo_matrix((np.ones(len(row)), (row, col)))

    return from_scipy_sparse_matrix(spm, True)

def test_node_subgraph_with_halo():
    gi = create_large_graph_index(1000)
    nodes = np.random.choice(gi.number_of_nodes(), 100, replace=False)
    halo_subg, inner_node, inner_edge = gi.node_halo_subgraph(toindex(nodes), 2)

    # Check if edges in the subgraph are in the original graph.
    for s, d, e in zip(*halo_subg.graph.edges()):
        assert halo_subg.induced_edges[e] in gi.edge_id(
                halo_subg.induced_nodes[s], halo_subg.induced_nodes[d])

    # Check if the inner node labels are correct.
    inner_node = inner_node.asnumpy()
    inner_node_ids = np.nonzero(inner_node)[0]
    inner_node_ids = halo_subg.induced_nodes.tonumpy()[inner_node_ids]
    assert np.all(inner_node_ids == np.sort(nodes))

    # Check if the inner edge labels are correct.
    inner_edge = inner_edge.asnumpy()
    inner_edge_ids = halo_subg.induced_edges.tonumpy()[inner_edge > 0]
    subg = gi.node_subgraph(toindex(nodes))
    assert np.all(np.sort(subg.induced_edges.tonumpy()) == np.sort(inner_edge_ids))

if __name__ == '__main__':
    test_node_subgraph()
    test_node_subgraph_with_halo()
    test_edge_subgraph()
    test_edge_subgraph_preserve_nodes()
    test_immutable_edge_subgraph()
    test_immutable_edge_subgraph_preserve_nodes()
