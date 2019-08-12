""" Build k-hop graph. """
from numpy.linalg import matrix_power

from ..graph import DGLGraph
from ..graph_index import from_csr


def khop_graph(graph, k):
    """
    Generate the graph that gathers k-hop information of the given graph.
    The adjacency matrix of the returned graph is A^k(where A is the adjacency matrix of g).
    """
    adj_new = graph.adjacency_matrix_scipy(fmt='csr', return_edge_ids=False) ** k
    return adj_new, DGLGraph(from_csr(adj_new.indptr, adj_new.indices, True, 'out'), readonly=True)
