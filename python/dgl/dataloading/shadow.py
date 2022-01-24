from ..frame import LazyFeature
from ..utils import EidExcluder

class ShaDowKHopSampler(object):
    """ShaDow-GNN Sampler."""
    def __init__(self, fanouts, prefetch_node_feats=None, prefetch_edge_feats=None,
                 output_device=None):
        self.fanouts = fanouts
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(self, g, seed_nodes, exclude_edges=None):
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, output_device=self.output_device,
                exclude_edges=exclude_edges)
            block = transform.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[dgl.NID]
        sg = g.subgraph(seed_nodes, relabel_nodes=True)
        sg = EidExcluder(exclude_edges)(sg)
        sg.ndata.update({k: LazyFeature(k) for k in self.prefetch_node_feats})
        sg.edata.update({k: LazyFeature(k) for k in self.prefetch_edge_feats})
        return sg
