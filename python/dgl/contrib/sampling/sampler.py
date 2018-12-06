# This file contains subgraph samplers.

import numpy as np

from ... import utils
from ...subgraph import DGLSubGraph
from ... import backend as F

__all__ = ['NeighborSampler']

class NSSubgraphLoader(object):
    def __init__(self, g, batch_size, expand_factor, num_hops=1,
                 neighbor_type='in', node_prob=None, seed_nodes=None,
                 shuffle=False, num_workers=1, max_subgraph_size=None,
                 return_seed_id=False):
        self._g = g
        if not g._graph.is_readonly():
            raise NotImplementedError("subgraph loader only support read-only graphs.")
        self._batch_size = batch_size
        self._expand_factor = expand_factor
        self._num_hops = num_hops
        self._node_prob = node_prob
        self._return_seed_id = return_seed_id
        if self._node_prob is not None:
            assert self._node_prob.shape[0] == g.number_of_nodes(), \
                    "We need to know the sampling probability of every node"
        if seed_nodes is None:
            self._seed_nodes = F.arange(0, g.number_of_nodes())
        else:
            self._seed_nodes = seed_nodes
        if shuffle:
            self._seed_nodes = F.rand_shuffle(self._seed_nodes)
        self._num_workers = num_workers
        if max_subgraph_size is None:
            # This size is set temporarily.
            self._max_subgraph_size = 1000000
        else:
            self._max_subgraph_size = max_subgraph_size
        self._neighbor_type = neighbor_type
        self._subgraphs = []
        self._seed_ids = []
        self._subgraph_idx = 0

    def _prefetch(self):
        seed_ids = []
        num_nodes = len(self._seed_nodes)
        for i in range(self._num_workers):
            start = self._subgraph_idx * self._batch_size
            # if we have visited all nodes, don't do anything.
            if start >= num_nodes:
                break
            end = min((self._subgraph_idx + 1) * self._batch_size, num_nodes)
            seed_ids.append(utils.toindex(self._seed_nodes[start:end]))
            self._subgraph_idx += 1
        sgi = self._g._graph.neighbor_sampling(seed_ids, self._expand_factor,
                                               self._num_hops, self._neighbor_type,
                                               self._node_prob, self._max_subgraph_size)
        subgraphs = [DGLSubGraph(self._g, i.induced_nodes, i.induced_edges, \
                i) for i in sgi]
        self._subgraphs.extend(subgraphs)
        if self._return_seed_id:
            self._seed_ids.extend(seed_ids)

    def __iter__(self):
        return self

    def __next__(self):
        # If we don't have prefetched subgraphs, let's prefetch them.
        if len(self._subgraphs) == 0:
            self._prefetch()
        # At this point, if we still don't have subgraphs, we must have
        # iterate all subgraphs and we should stop the iterator now.
        if len(self._subgraphs) == 0:
            raise StopIteration
        aux_infos = {}
        if self._return_seed_id:
            aux_infos['seeds'] = self._seed_ids.pop(0).tousertensor()
        return self._subgraphs.pop(0), aux_infos

def NeighborSampler(g, batch_size, expand_factor, num_hops=1,
                    neighbor_type='in', node_prob=None, seed_nodes=None,
                    shuffle=False, num_workers=1, max_subgraph_size=None,
                    return_seed_id=False):
    '''Create a sampler that samples neighborhood.

    .. note:: This method currently only supports MXNet backend. Set
        "DGLBACKEND" environment variable to "mxnet".

    This creates a subgraph data loader that samples subgraphs from the input graph
    with neighbor sampling. This simpling method is implemented in C and can perform
    sampling very efficiently.
    
    A subgraph grows from a seed vertex. It contains sampled neighbors
    of the seed vertex as well as the edges that connect neighbor nodes with
    seed nodes. When the number of hops is k (>1), the neighbors are sampled
    from the k-hop neighborhood. In this case, the sampled edges are the ones
    that connect the source nodes and the sampled neighbor nodes of the source
    nodes.

    The subgraph loader returns a list of subgraphs and a dictionary of additional
    information about the subgraphs. The size of the subgraph list is the number of workers.
    The dictionary contains:
        'seeds': a list of 1D tensors of seed Ids, if return_seed_id is True.

    Parameters
    ----------
    g: the DGLGraph where we sample subgraphs.
    batch_size: The number of subgraphs in a batch.
    expand_factor: the number of neighbors sampled from the neighbor list
        of a vertex. The value of this parameter can be
        an integer: indicates the number of neighbors sampled from a neighbor list.
        a floating-point: indicates the ratio of the sampled neighbors in a neighbor list.
        string: indicates some common ways of calculating the number of sampled neighbors,
        e.g., 'sqrt(deg)'.
    num_hops: The size of the neighborhood where we sample vertices.
    neighbor_type: indicates the neighbors on different types of edges.
        "in" means the neighbors on the in-edges, "out" means the neighbors on
        the out-edges and "both" means neighbors on both types of edges.
    node_prob: the probability that a neighbor node is sampled.
        1D Tensor. None means uniform sampling. Otherwise, the number of elements
        should be the same as the number of vertices in the graph.
    seed_nodes: a list of nodes where we sample subgraphs from.
        If it's None, the seed vertices are all vertices in the graph.
    shuffle: indicates the sampled subgraphs are shuffled.
    num_workers: the number of worker threads that sample subgraphs in parallel.
    max_subgraph_size: the maximal subgraph size in terms of the number of nodes.
        GPU doesn't support very large subgraphs.
    return_seed_id: indicates whether to return seed ids along with the subgraphs.
        The seed Ids are in the parent graph.
    
    Returns
    -------
    A subgraph loader that returns a list of batched subgraphs and a dictionary of
        additional information about the subgraphs.
    '''
    return NSSubgraphLoader(g, batch_size, expand_factor, num_hops, neighbor_type, node_prob,
                            seed_nodes, shuffle, num_workers, max_subgraph_size, return_seed_id)
