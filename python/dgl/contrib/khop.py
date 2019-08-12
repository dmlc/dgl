""" Build k-hop graph. """

from ..graph import DGLGraph
from ..graph_index import from_coo
from numba import jit
import numpy as np


@jit(nopython=True)
def duplicate(arr, times):
    n = len(arr)
    lengths = 0
    for i in range(n):
        lengths += times[i]

    rst = np.empty(shape=(lengths), dtype=np.int64)
    cnt = 0
    for i in range(n):
        for j in range(times[i]):
            rst[cnt] = arr[i]
            cnt += 1
    return rst

def khop_graph(graph, k):
    """
    Generate the graph that gathers k-hop information of the given graph.
    The adjacency matrix of the returned graph is A^k(where A is the adjacency matrix of g).
    """
    n = graph.number_of_nodes()
    adj_new = graph.adjacency_matrix_scipy(return_edge_ids=False) ** k
    adj_new = adj_new.tocoo()
    data = adj_new.data
    row = duplicate(adj_new.row, data)
    col = duplicate(adj_new.col, data)
    return DGLGraph(from_coo(n, row, col, True, True))
