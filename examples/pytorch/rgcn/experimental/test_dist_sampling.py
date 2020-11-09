import dgl
import torch as th
import numpy as np
from dgl.distributed import DistDataLoader

from partition_graph import load_ogb

ip_config = 'ip_config.txt'
num_servers = 1
num_workers = 0
graph_name = 'ogbn-mag'
part_config = 'data/ogbn-mag.json'
dataset = 'ogbn-mag'

batch_size = 1000
fanouts=[10,25]

gpb, _, ntypes, etypes = dgl.distributed.load_partition_book(part_config, 0)
num_parts = gpb.num_partitions()

# Load the original graph.
print('load', dataset)
orig_g = load_ogb(dataset, False)
etype_map = {orig_g.get_etype_id(etype):etype for etype in orig_g.etypes}
etype_to_eptype = {orig_g.get_etype_id(etype):(src_ntype, dst_ntype) for src_ntype, etype, dst_ntype in orig_g.canonical_etypes}

print('load partitions of', dataset)
parts = []
for i in range(num_parts):
    part, node_feat, edge_feat, gpb, _, ntypes, etypes = dgl.distributed.load_partition(part_config, i)
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
train_nid = dgl.distributed.node_split(g.nodes['paper'].data['train_mask'], pb, ntype='paper', force_even=True)
val_nid = dgl.distributed.node_split(g.nodes['paper'].data['val_mask'], pb, ntype='paper', force_even=True)
test_nid = dgl.distributed.node_split(g.nodes['paper'].data['test_mask'], pb, ntype='paper', force_even=True)
labels = g.nodes['paper'].data['labels'][np.arange(g.number_of_nodes('paper'))]
homo_train_nid = gpb.map_to_homo_nid(train_nid, 'paper')
homo_val_nid = gpb.map_to_homo_nid(val_nid, 'paper')
homo_test_nid = gpb.map_to_homo_nid(test_nid, 'paper')
train_ntype, train_nid1 = gpb.map_to_per_ntype(homo_train_nid)
assert np.all((train_ntype == train_ntype[0]).numpy())
assert np.all((train_nid1 == train_nid).numpy())
train_nid = homo_train_nid
val_nid = homo_val_nid
test_nid = homo_test_nid

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
        etypes = []
        norms = []
        ntypes = []
        seeds = th.LongTensor(np.asarray(seeds))
        cur = seeds
        for fanout in self.fanouts:
            frontier = self.sample_neighbors(self.g, cur, fanout, replace=True)
            block = dgl.to_block(frontier, cur)

            block.edata[dgl.EID] = frontier.edata[dgl.EID]
            block.edata[dgl.ETYPE] = self.g.map_to_per_etype(frontier.edata[dgl.EID])[0]
            block.srcdata[dgl.NTYPE] = self.g.map_to_per_ntype(block.srcdata[dgl.NID])[0]
            block.dstdata[dgl.NTYPE] = self.g.map_to_per_ntype(block.dstdata[dgl.NID])[0]
            cur = block.srcdata[dgl.NID]
            #blocks.insert(0, block)
            blocks.insert(0, block)
        return blocks

print('create sampler')
sampler = NeighborSampler(g, fanouts, dgl.distributed.sample_neighbors)
# Create DataLoader for constructing blocks
# TODO We need to convert train node Ids to homogeneous Ids.
dataloader = DistDataLoader(
        dataset=test_nid,
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
        # Get node/edge types.
        src_type = block.srcdata[dgl.NTYPE][src].numpy()
        dst_type = block.dstdata[dgl.NTYPE][dst].numpy()
        etype = block.edata[dgl.ETYPE][eid].numpy()
        # These are global Ids in the original graph.
        orig_src = orig_nid_map[shuffled_src]
        orig_dst = orig_nid_map[shuffled_dst]
        orig_eid = orig_eid_map[shuffled_eid]

        for e in np.unique(etype):
            src_t = src_type[etype == e]
            dst_t = dst_type[etype == e]
            assert np.all(src_t == src_t[0])
            assert np.all(dst_t == dst_t[0])

            # Check the node Ids and edge Ids.
            orig_src1, orig_dst1 = orig_g.find_edges(orig_eid[etype == e], etype=etype_map[e])
            orig_src1 = orig_src1.numpy()
            orig_dst1 = orig_dst1.numpy()
            assert np.all(orig_src1 == orig_src[etype == e])
            assert np.all(orig_dst1 == orig_dst[etype == e])

            # Check the node types.
            src_ntype, dst_ntype = etype_to_eptype[e]
            assert np.all(src_t == orig_g.get_ntype_id(src_ntype))
            assert np.all(dst_t == orig_g.get_ntype_id(dst_ntype))

            # Check node features.
            # Map the global node Ids to per-type node Ids after shuffling.
            ntype_id, typed_src = gpb.map_to_per_ntype(shuffled_src[etype == e])
            assert np.all(ntype_id.numpy() == src_t)
            ntype_id, typed_dst = gpb.map_to_per_ntype(shuffled_dst[etype == e])
            assert np.all(ntype_id.numpy() == dst_t)
            if src_ntype in orig_g.ntypes:
                for feat_name in orig_g.nodes[src_ntype].data:
                    assert np.all(g.nodes[src_ntype].data[feat_name][typed_src].numpy() == orig_g.nodes[src_ntype].data[feat_name][orig_src1].numpy())
            if dst_ntype in orig_g.ntypes:
                for feat_name in orig_g.nodes[dst_ntype].data:
                    assert np.all(g.nodes[dst_ntype].data[feat_name][typed_dst].numpy() == orig_g.nodes[dst_ntype].data[feat_name][orig_dst1].numpy())
