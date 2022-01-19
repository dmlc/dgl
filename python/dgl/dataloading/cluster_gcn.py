import os
import pickle
import numpy as np

from .. import backend as F
from ..base import DGLError
from ..partition import metis_partition_assignment
from ..frame import LazyFeature

class ClusterGCNSampler(object):
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
        node_ids = F.cat([
            self.partition_node_ids[self.partition_offset[i]:self.partition_offset[i+1]]
            for i in F.asnumpy(partition_ids)], 0)
        sg = g.subgraph(node_ids, relabel_nodes=True, output_device=self.output_device)
        sg.ndata.update({k: LazyFeature(k) for k in self.prefetch_node_feats})
        sg.edata.update({k: LazyFeature(k) for k in self.prefetch_edge_feats})
        return sg
