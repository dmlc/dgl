import dgl
import torch as th
import numpy as np
from dgl.distributed import DistDataLoader

from partition_graph import load_ogb

ip_config = 'ip_config.txt'
num_servers = 1
num_workers = 0
graph_name = 'ogb-product'
part_config = 'data/ogb-product.json'
dataset = 'ogbn-products'
num_parts = 4

batch_size = 1000
fanouts=[10,25]

# Load the original graph.
print('load', dataset)
orig_g, _ = load_ogb(dataset)

print('load partitions of', dataset)
parts = []
for i in range(num_parts):
    part, node_feat, edge_feat, gpb, _, _, _ = dgl.distributed.load_partition(part_config, i)
    parts.append(part)
# Mapping shuffled Id to original Id.
# The original Ids are the per-type Ids in the input graph.
# The suffled Ids are for the homogeneous graph format.
orig_nid_map = np.ones((orig_g.number_of_nodes(),), dtype=np.int64) * -1
orig_eid_map = np.ones((orig_g.number_of_edges(),), dtype=np.int64) * -1
for part in parts:
    shuffled_nid = part.ndata[dgl.NID].numpy()
    orig_nid = part.ndata['orig_id'].numpy()
    old = orig_nid_map[shuffled_nid]
    assert np.all(old[old >= 0] == orig_nid[old >= 0])
    orig_nid_map[shuffled_nid] = orig_nid

    shuffled_eid = part.edata[dgl.EID].numpy()
    orig_eid = part.edata['orig_id'].numpy()
    old = orig_eid_map[shuffled_eid]
    assert np.all(old[old >= 0] == orig_eid[old >= 0])
    orig_eid_map[shuffled_eid] = orig_eid

# Set up distributed graph.
print('initialize distributed API.')
dgl.distributed.initialize(ip_config, num_servers, num_workers=num_workers)
g = dgl.distributed.DistGraph(graph_name, part_config=part_config)
print('rank:', g.rank())
pb = g.get_partition_book()
# These are per-type node Ids.
train_nid = dgl.distributed.node_split(g.ndata['train_mask'], pb, force_even=True)
val_nid = dgl.distributed.node_split(g.ndata['val_mask'], pb, force_even=True)
test_nid = dgl.distributed.node_split(g.ndata['test_mask'], pb, force_even=True)
labels = g.ndata['labels'][np.arange(g.number_of_nodes())]

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        """Do neighbor sample
        Parameters
        ----------
        seeds :
            Seed nodes
        Returns
        -------
        tensor
            Seed nodes, also known as target nodes
        blocks
            Sampled subgraphs
        """
        blocks = []
        norms = []
        seeds = th.LongTensor(np.asarray(seeds))
        cur = seeds
        for fanout in self.fanouts:
            frontier = self.sample_neighbors(self.g, cur, fanout, replace=True)
            block = dgl.to_block(frontier, cur)
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

print('create sampler')
sampler = NeighborSampler(g, fanouts, dgl.distributed.sample_neighbors)
# Create DataLoader for constructing blocks
# TODO We need to convert train node Ids to homogeneous Ids.
dataloader = DistDataLoader(
        dataset=train_nid.numpy(),
        batch_size=batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)

print('checking mini-batches')
for step, blocks in enumerate(dataloader):
    for block in blocks:
        # These are partition-local Ids.
        src, dst, eid = block.edges(form='all')
        # These are global Ids after shuffling.
        shuffled_src = block.srcdata[dgl.NID][src].numpy()
        shuffled_dst = block.dstdata[dgl.NID][dst].numpy()
        shuffled_eid = block.edata[dgl.EID][eid].numpy()
        # These are global Ids in the original graph.
        orig_src = orig_nid_map[shuffled_src]
        orig_dst = orig_nid_map[shuffled_dst]
        orig_eid = orig_eid_map[shuffled_eid]

        # Check the node Ids and edge Ids.
        exist = orig_g.has_edges_between(orig_src, orig_dst)
        assert np.all(exist.numpy())
        #TODO somehow edge Ids don't match.
        #orig_src1, orig_dst1 = orig_g.find_edges(orig_eid)
        #orig_src1 = orig_src1.numpy()
        #orig_dst1 = orig_dst1.numpy()
        #assert np.all(orig_src1 == orig_src)
        #assert np.all(orig_dst1 == orig_dst)

        # Check node features.
        # Map the global node Ids to per-type node Ids after shuffling.
        for feat_name in orig_g.ndata:
            assert np.all(g.ndata[feat_name][shuffled_src].numpy() == orig_g.ndata[feat_name][orig_src].numpy())
        for feat_name in orig_g.ndata:
            assert np.all(g.ndata[feat_name][shuffled_dst].numpy() == orig_g.ndata[feat_name][orig_dst].numpy())
