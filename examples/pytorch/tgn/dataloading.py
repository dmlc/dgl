import dgl
import torch
from dgl.dataloading.dataloader import EdgeCollator,_prepare_tensor,_pop_subgraph_storage,_pop_blocks_storage,assign_block_eids
from dgl.dataloading import BlockSampler
from dgl import transform
from functools import partial


class DictNode:
    def __init__(self,parent,NIDdict=None):
        self.parent = parent
        if NIDdict != None:
            self.NIDdict = NIDdict.numpy()
        else:
            self.NIDdict = None 
        if (parent!=None and NIDdict==None) or (parent == None and NIDdict!=None):
            raise ValueError("Parent and Dict Unmatched")

    def GetRootID(self,index):
        if isinstance(index,torch.tensor):
            index = index.numpy()
        if self.parent==None:
            return index
        map_index = self.NIDdict[index]
        return self.parent.GetRootID(map_index)

    def GetAncestorID(self,index,level=None):
        if isinstance(index,torch.tensor):
            index = index.numpy()
        if self.parent == None or level==0:
            return index
        map_index = self.NIDdict[index]
        parent_level = None if level== None else level-1
        return self.parent.GetAncestorID(map_index,parent_level)


class NegativeSampler:
    def __init__(self,g,k):
        self.weights = torch.arange(g.num_nodes())
        self.k = k
    
    def __call__(self,g,eids):
        src,_ = g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src),replacement=True)
        return src,dst


# Will it cause some id mapping issue?
class TemporalSampler(BlockSampler):
    def __init__(self,sampler_type='topk',k=10):
        super().__init__(1)
        if sampler_type == 'topk':
            self.sampler = partial(dgl.sampling.select_topk(),k=k)
        else:
            self.sampler = partial(dgl.sampling.sample_neighbors,fanout=k)
        
        
    # Here Assume seed_nodes' timestamps have been updated
    def sampler_frontier(self,
                        block_id,
                        g,
                        seed_nodes,
                        timestamp):
        full_neighbor_subgraph = dgl.in_subgraph(g,seed_nodes)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes,seed_nodes)

        temporal_edge_mask = full_neighbor_subgraph.edata['timestamp'] < timestamp
        temporal_subgraph = dgl.edge_subgraph(full_neighbor_subgraph,temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        # The added new edgge will be preserved hence 
        root2sub_dict = dict(zip(temp2origin.tolist(),temporal_subgraph.nodes().tolist()))
        seed_nodes = [root2sub_dict[n] for n in seed_nodes] 
        final_subgraph = self.sampler(g=temporal_subgraph,nodes=seed_nodes)
        final_subgraph.remove_self_loop()
        return final_subgraph

        # Temporal Subgraph
    def sample_blocks(self,
                      g,
                      seed_nodes,
                      timestamp):
        blocks = []
        frontier = self.sampler_frontier(0,g,seed_nodes,timestamp)
        block = transform.to_block(frontier,seed_nodes)
        if self.return_eids:
            assign_block_eids(block,frontier)
        blocks.append(block)
        return blocks


        
# Assuming only handle with negative sampler
# Assume the order of items is chronological
# Assume items is a tensor of ID instead lf
# Assume g sampling is the same as g sa
class TemporalEdgeCollator(EdgeCollator):
    def _collate_with_negative_sampling(self,items):
        # Here items are edge id
        items = _prepare_tensor(self.g_sampling,items,'items',self.is_distributed)
        # Here node id will not change
        pair_graph = self.g.edge_subgraph(items,preserve_nodes = True)
        induced_edges = pair_graph.edata[dgl.EID]

        neg_srcdst_raw = self.negative_sampler(self.g,items)
        neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst_raw}
        dtype = torch.dtype(list(neg_srcdst.values())[0][0])
        neg_edges = {
            etype: neg_srcdst.get(etype, (torch.tensor([], dtype), torch.tensor([], dtype)))
            for etype in self.g.canonical_etypes}

        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        pair_graph, neg_pair_graph = dgl.transform.compact_graph([pair_graph,neg_pair_graph])
        # Need to remap id

        pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][pair_graph.ndata[dgl.NID]]
        neg_pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][neg_pair_graph.ndata[dgl.NID]]

        pair_graph.edata[dgl.EID] = induced_edges
        
        #seed_nodes = pair_graph.ndata[dgl.NID]
        batch_graphs = []
        nodes_id = []
        timestamps = []
        # Will not work since node id is batched graph
        for src,dst in zip(self.g_sampling.find_edges(items)):
            ts = pair_graph.edata['timestamp']
            timestamps.append(ts)
            subg = self.block_sampler.sample_blocks(self.g_sampling,
                                                    [src,dst],
                                                    timestamp=ts)[0]
            # Result Graph will be bipartite
            # here the nid will be?
            nodes_id.append(subg.srcdata[dgl.NID])
            batch_graphs.append(subg)
        timestamps = torch.tensor(timestamps).repeat_interleave(self.negative_sampler.k)
        for i,neg_edge in enumerate(zip(neg_srcdst_raw)):
            ts = timestamps[i]
            subg = self.block_sampler.sample_blocks(self.g_sampling,
                                                    [neg_edge[1]],
                                                    timestamp=ts)[0]
            nodes_id.append(subg.ndata[dgl.NID][subg.edges()[1]])
            batch_graphs.append(subg)
        blocks = [dgl.batch(batch_graphs)]
        input_nodes = torch.cat(nodes_id)
        return input_nodes,pair_graph,neg_pair_graph,blocks
        
    def collator(self,items):
        result = super().collate(items)
        # Copy the feature from parent graph
        _pop_subgraph_storage(result[1],self.g)
        _pop_subgraph_storage(result[2],self.g)
        _pop_blocks_storage(result[-1],self.g_sampling)
        return result
            

# use Edgedataloader, generate block whose NID is the root
# Can handle new node but need further projection during the test new node phase
# Output is two graphs
# The embedding is computed on blocks,
# The feature is z will be mapped to pair_neg_graph and pair_graph
# The link will be predicted via message passing
# Positive link graph can be also used for memory update

# ====== Potential Issues ======
# ID mapping is most likely to be problematic
# We can use node wise feature: Like  secondary ID for implementation validation



        

        