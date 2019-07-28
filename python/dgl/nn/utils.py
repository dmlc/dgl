"""Graph construction utilities used in dgl.nn modules"""

import numpy as np

from ..graph import DGLGraph
from ..batched_graph import BatchedDGLGraph, batch
from ..graph_index import from_csr
from .. import backend as F


__all__ = ['_create_fully_connected_graph', '_create_bipartite_graph',
           '_create_batched_graph_from_num_nodes']

def _create_fully_connected_graph(graph):
    r""" Create a fully connected graph upon the given graph."""
    if isinstance(graph, BatchedDGLGraph):
        gs = []
        for n_nodes in graph.batch_num_nodes:
            indptr = np.arange(0, n_nodes * n_nodes + 1, n_nodes)
            indices = np.tile(np.arange(n_nodes), n_nodes)
            gs.append(
                DGLGraph(from_csr(indptr, indices, False, 'out'), readonly=True)
            )
        return batch(gs)
    else:
        n_nodes = graph.number_of_nodes()
        indptr = np.arange(0, n_nodes * n_nodes + 1, n_nodes)
        indices = np.tile(np.arange(n_nodes), n_nodes)
        return DGLGraph(from_csr(indptr, indices, False, 'out'), readonly=True)

def _create_bipartite_graph(graph_u, graph_v):
    r""" Create a bipartite G(U, V, E) graph with nodes in graph_u as U and nodes in graph_v as V"""
    if isinstance(graph_u, BatchedDGLGraph):
        gs = []
        if graph_u.batch_size != graph_v.batch_size:
            raise KeyError('Batch size of graph_u does not equal that of graph_v.')
        us, vs = [], []
        v_shift = 0
        for n_nodes_u, n_nodes_v in zip(graph_u.batch_num_nodes, graph_v.batch_num_nodes):
            indptr = np.concatenate([
                np.arange(0, n_nodes_u * n_nodes_v + 1, n_nodes_v),
                np.full(n_nodes_v, n_nodes_u * n_nodes_v, dtype=np.int64)
            ])
            indices = np.tile(np.arange(n_nodes_u, n_nodes_u + n_nodes_v), n_nodes_u)
            gs.append(
                DGLGraph(from_csr(indptr, indices, False, 'out'), readonly=True)
            )
            us.append(np.arange(v_shift, v_shift + n_nodes_u))
            vs.append(np.arange(v_shift + n_nodes_u, v_shift + n_nodes_u + n_nodes_v))
            v_shift += n_nodes_u + n_nodes_v
        return batch(gs),\
               F.zerocopy_from_numpy(np.concatenate(us)),\
               F.zerocopy_from_numpy(np.concatenate(vs))
    else:
        n_nodes_u = graph_u.number_of_nodes()
        n_nodes_v = graph_v.number_of_nodes()
        us, vs = np.arange(n_nodes_u), np.arange(n_nodes_u, n_nodes_u + n_nodes_v)
        indptr = np.concatenate([
            np.arange(0, n_nodes_u * n_nodes_v + 1, n_nodes_v),
            np.full(n_nodes_v, n_nodes_u * n_nodes_v, )
        ])
        indices = np.tile(np.arange(n_nodes_u, n_nodes_u + n_nodes_v), n_nodes_u)
        return DGLGraph(from_csr(indptr, indices, False, 'out'), readonly=True), \
            F.zerocopy_from_numpy(us), F.zerocopy_from_numpy(vs)

def _create_batched_graph_from_num_nodes(n_nodes_arr):
    r""" Create a batched graph (with no nodes) from number of nodes list
    to use as the latent graph in GNN layers."""
    gs = []
    for n_nodes in n_nodes_arr:
        indptr = np.zeros(n_nodes + 1, dtype=np.int64)
        indices = np.array([])
        gs.append(
            DGLGraph(from_csr(indptr, indices, False, 'out'), readonly=True)
        )
    if len(gs) == 0:
        return gs[0]
    return batch(gs)

