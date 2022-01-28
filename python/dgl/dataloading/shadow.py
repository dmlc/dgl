from ..frame import LazyFeature
from ..sampling.utils import EidExcluder
from .. import transform
from ..base import NID
from .base import set_node_lazy_features, set_edge_lazy_features

class ShaDowKHopSampler(object):
    """ShaDow-GNN Sampler."""
    def __init__(self, fanouts, prefetch_node_feats=None, prefetch_edge_feats=None,
                 output_device=None):
        self.fanouts = fanouts
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(self, g, seed_nodes, exclude_edges=None):
        output_nodes = seed_nodes
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, output_device=self.output_device,
                exclude_edges=exclude_edges)
            block = transform.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[NID]

        sg = g.subgraph(seed_nodes, relabel_nodes=True, output_device=self.output_device)
        if exclude_edges is not None:
            sg = EidExcluder(exclude_edges)(sg)

        set_node_lazy_features(sg, self.prefetch_node_feats)
        set_edge_lazy_features(sg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, sg
