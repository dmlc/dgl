import dgl
from dgl.frame import LazyFeature
from .base import BlockSampler

class NeighborSampler(BlockSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False, **kwargs):
        super().__init__(**kwargs)
        self.fanouts = fanouts
        self.edge_dir = edge_dir
        self.prob = prob
        self.replace = replace

    def sample_blocks(self, g, seed_nodes, exclude_edges=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_edges)
            eid = frontier.edata[dgl.EID]
            block = dgl.to_block(frontier, seed_nodes)
            block.edata[dgl.EID] = eid
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)

        return seed_nodes, output_nodes, blocks
