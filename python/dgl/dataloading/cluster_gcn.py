"""Cluster-GCN samplers."""
import os
import pickle
import numpy as np

from .. import backend as F
from ..base import DGLError
from ..partition import metis_partition_assignment
from .base import set_node_lazy_features, set_edge_lazy_features

class ClusterGCNSampler(object):
    """Cluster-GCN sampler.

    This sampler first partitions the graph with METIS partitioning, then it caches the nodes of
    each partition to a file within the given cache directory.

    This is used in conjunction with :class:`dgl.dataloading.DataLoader`.

    Notes
    -----
    The graph must be homogeneous and on CPU.

    Parameters
    ----------
    g : DGLGraph
        The original graph.
    k : int
        The number of partitions.
    cache_path : str
        The path to the cache directory for storing the partition result.
    """
    def __init__(self, g, k, balance_ntypes=None, balance_edges=False, mode='k-way',
                 prefetch_node_feats=None, prefetch_edge_feats=None, output_device=None,
                 cache_path='cluster_gcn.pkl'):
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    self.partition_offset, self.partition_node_ids = pickle.load(f)
            except (EOFError, TypeError, ValueError):
                raise DGLError(
                    f'The contents in the cache file {cache_path} is invalid. '
                    f'Please remove the cache file {cache_path} or specify another path.')
            if len(self.partition_offset) != k + 1:
                raise DGLError(
                    f'Number of partitions in the cache does not match the value of k. '
                    f'Please remove the cache file {cache_path} or specify another path.')
            if len(self.partition_node_ids) != g.num_nodes():
                raise DGLError(
                    f'Number of nodes in the cache does not match the given graph. '
                    f'Please remove the cache file {cache_path} or specify another path.')
        else:
            partition_ids = metis_partition_assignment(
                g, k, balance_ntypes=balance_ntypes, balance_edges=balance_edges, mode=mode)
            partition_ids = F.asnumpy(partition_ids)
            partition_node_ids = np.argsort(partition_ids)
            partition_size = F.zerocopy_from_numpy(np.bincount(partition_ids, minlength=k))
            partition_offset = F.zerocopy_from_numpy(np.insert(np.cumsum(partition_size), 0, 0))
            partition_node_ids = F.zerocopy_from_numpy(partition_ids)
            with open(cache_path, 'wb') as f:
                pickle.dump((partition_offset, partition_node_ids), f)
            self.partition_offset = partition_offset
            self.partition_node_ids = partition_node_ids

        self.prefetch_node_feats = prefetch_node_feats or []
        self.prefetch_edge_feats = prefetch_edge_feats or []
        self.output_device = output_device

    def sample(self, g, partition_ids):
        """Samples a subgraph given a list of partition IDs."""
        node_ids = F.cat([
            self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i+1]]
            for i in F.asnumpy(partition_ids)], 0)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        set_node_lazy_features(sg, self.prefetch_node_feats)
        set_edge_lazy_features(sg, self.prefetch_edge_feats)
        return sg
