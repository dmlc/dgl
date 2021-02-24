import dgl
import torch
from dgl.dataloading.dataloader import EdgeCollator, assign_block_eids
from dgl.dataloading import BlockSampler
from dgl.dataloading.pytorch import _pop_subgraph_storage, _pop_blocks_storage
from dgl import transform
from termcolor import colored
from functools import partial
import copy


def _prepare_tensor(g,data,name,is_distributed):
    return torch.tensor(data) if is_distributed else dgl.utils.prepare_tensor(g,data,name)

# Sort the graph edge list by timestamp
def temporal_sort(g,key):
    edge_keys = list(g.edata.keys())
    node_keys = list(g.ndata.keys())

    sorted_idx = g.edata[key].sort()[1]
    buf_graph = dgl.graph((g.edges()[0][sorted_idx],g.edges()[1][sorted_idx]))
    # copy back edge and node data
    for ek in edge_keys:
        buf_graph.edata[ek] = g.edata[ek][sorted_idx]

    # Since node index unchanged direct copy
    for nk in node_keys:
        buf_graph.ndata[nk] = g.ndata[nk]
    return buf_graph

# Uniformly sampler neighboring edges
class NegativeSampler:
    def __init__(self,g,k):
        self.weights = torch.ones(g.num_nodes()).float()/g.num_nodes()
        self.k = k
    
    def __call__(self,g,eids):
        src,_ = g.find_edges(eids)
        src = src.repeat_interleave(self.k)
        dst = self.weights.multinomial(len(src),replacement=True)
        return src,dst

class TemporalSampler(BlockSampler):
    def __init__(self,sampler_type='topk',k=10):
        super(TemporalSampler,self).__init__(1,False)
        if sampler_type == 'topk':
            self.sampler = partial(dgl.sampling.select_topk,k=k,weight='timestamp')
        else:
            self.sampler = partial(dgl.sampling.sample_neighbors,fanout=k)
        
        
    # Here Assume seed_nodes' timestamps have been updated
    # g is the bidirection masked graph for training
    # Seed nodes come from uni-directional 
    def sampler_frontier(self,
                        block_id,
                        g,
                        seed_nodes,
                        timestamp):
        full_neighbor_subgraph = dgl.in_subgraph(g,seed_nodes)
        full_neighbor_subgraph = dgl.add_edges(full_neighbor_subgraph,
                                               seed_nodes,seed_nodes)

        temporal_edge_mask = (full_neighbor_subgraph.edata['timestamp'] < timestamp) + (full_neighbor_subgraph.edata['timestamp'] <=0)
        temporal_subgraph = dgl.edge_subgraph(full_neighbor_subgraph,temporal_edge_mask)

        # Map preserve ID
        temp2origin = temporal_subgraph.ndata[dgl.NID]
        
        # The added new edgge will be preserved hence 
        root2sub_dict = dict(zip(temp2origin.tolist(),temporal_subgraph.nodes().tolist()))
        temporal_subgraph.ndata[dgl.NID] = g.ndata[dgl.NID][temp2origin]
        seed_nodes = [root2sub_dict[int(n)] for n in seed_nodes] 
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
        #block = transform.to_block(frontier,seed_nodes)
        block = frontier
        if self.return_eids:
            assign_block_eids(block,frontier)
        blocks.append(block)
        return blocks

class TemporalEdgeCollator(EdgeCollator):
    def _collate_with_negative_sampling(self,items):
        items = _prepare_tensor(self.g_sampling,items,'items',False)
        # Here node id will not change
        pair_graph = self.g.edge_subgraph(items,preserve_nodes = True)
        induced_edges = pair_graph.edata[dgl.EID]

        neg_srcdst_raw = self.negative_sampler(self.g,items)
        neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst_raw}
        dtype = list(neg_srcdst.values())[0][0].dtype
        neg_edges = {
            etype: neg_srcdst.get(etype, (torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype)))
            for etype in self.g.canonical_etypes}

        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        pair_graph, neg_pair_graph = dgl.transform.compact_graphs([pair_graph,neg_pair_graph])
        # Need to remap id
        pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][pair_graph.ndata[dgl.NID]]
        neg_pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][neg_pair_graph.ndata[dgl.NID]]

        pair_graph.edata[dgl.EID] = induced_edges
        
        batch_graphs = []
        nodes_id = []
        timestamps = []

        for i,edge in enumerate(zip(self.g.edges()[0][items],self.g.edges()[1][items])):
            ts = pair_graph.edata['timestamp'][i]
            timestamps.append(ts)
            subg = self.block_sampler.sample_blocks(self.g_sampling,
                                                    list(edge),
                                                    timestamp=ts)[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
            nodes_id.append(subg.srcdata[dgl.NID])
            batch_graphs.append(subg)
        timestamps = torch.tensor(timestamps).repeat_interleave(self.negative_sampler.k)
        for i,neg_edge in enumerate(zip(neg_srcdst_raw[0].tolist(),neg_srcdst_raw[1].tolist())):
            ts = timestamps[i]
            subg = self.block_sampler.sample_blocks(self.g_sampling,
                                                    [neg_edge[1]],
                                                    timestamp=ts)[0]
            subg.ndata['timestamp'] = ts.repeat(subg.num_nodes())
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

class TemporalEdgeDataLoader(dgl.dataloading.EdgeDataLoader):
    def __init__(self,g,eids,block_sampler,device='cpu',collator=TemporalEdgeCollator,**kwargs):
        collator_kwargs = {}
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            if k in self.collator_arglist:
                collator_kwargs[k] = v
            else:
                dataloader_kwargs[k] = v
        self.collator = collator(g, eids, block_sampler, **collator_kwargs)

        assert not isinstance(g, dgl.distributed.DistGraph), \
                'EdgeDataLoader does not support DistGraph for now. ' \
                + 'Please use DistDataLoader directly.'
        self.dataloader = torch.utils.data.DataLoader(
            self.collator.dataset, collate_fn=self.collator.collate, **dataloader_kwargs)
        self.device = device

        # Precompute the CSR and CSC representations so each subprocess does not
        # duplicate.
        if dataloader_kwargs.get('num_workers', 0) > 0:
            g.create_formats_()

# ====== Fast Mode ======
class FastTemporalSampler:
    def __init__(self,g,k,device=torch.device('cpu')):
        self.k = k
        self.g = g
        num_nodes = g.num_nodes()
        self.neighbors = torch.empty((num_nodes,k),dtype=torch.long,device = device)
        self.e_id      = torch.empty((num_nodes,k),dtype=torch.long,device = device)
        self.__assoc__ = torch.empty(num_nodes,dtype=torch.long,device = device)
        self.last_update = torch.zeros(num_nodes,dtype=torch.double)
        self.reset()

    def sample_frontier(self,
                        block_id,
                        g,
                        seed_nodes):
        n_id = seed_nodes
        # Here Assume n_id is the bg nid
        neighbors = self.neighbors[n_id]
        nodes = n_id.view(-1,1).repeat(1,self.k)
        e_id  = self.e_id[n_id]
        mask  = e_id >= 0
        neighbors[~mask]=nodes[~mask]
        e_id = e_id[mask]
        neighbors = neighbors.flatten()
        nodes = nodes.flatten()
        n_id = torch.cat([n_id,neighbors]).unique()
        self.__assoc__[n_id] = torch.arange(n_id.size(0),device=n_id.device)
        neighbors, nodes = self.__assoc__[neighbors],self.__assoc__[nodes]
        subg = dgl.graph((nodes,neighbors))
        
        # The null feature for orphan node is zero
        subg.edata['timestamp'] = torch.zeros(subg.num_edges()).double()
        subg.edata['timestamp'][mask.flatten()] = self.g.edata['timestamp'][e_id]
        subg.ndata['timestamp'] = self.last_update[n_id]
        subg.edata['feats']     = torch.zeros((subg.num_edges(),self.g.edata['feats'].shape[1])).float()
        subg.edata['feats'][mask.flatten()]     = self.g.edata['feats'][e_id]
        
        #print(colored("Before : {}".format(subg.ndata[dgl.NID]),'red'))
        subg = dgl.remove_self_loop(subg)
        #print(colored("After: {}".format(subg.ndata[dgl.NID]),'blue'))
        subg = dgl.add_self_loop(subg)
        subg.ndata[dgl.NID] = n_id # From the root
        # Check self includeness
        
        for node in seed_nodes:
            if node not in subg.ndata[dgl.NID]:
                print(colored("Orphan Node Detected: {}".format(node),"green"))
                print(colored("Current seed n_id: {}".format(n_id),"red"))
                
        return subg

    def sample_blocks(self,
                      g,
                      seed_nodes):
        blocks = []
        frontier = self.sample_frontier(0,g,seed_nodes)
        block = frontier
        blocks.append(block)
        return blocks

    def add_edges(self,src,dst):
        neighbors = torch.cat([src, dst], dim=0)
        nodes = torch.cat([dst, src], dim=0)
        e_id = torch.arange(self.cur_e_id, self.cur_e_id + src.size(0),
                            device=src.device).repeat(2)
        self.cur_e_id += src.numel()

        # Convert newly encountered interaction ids so that they point to
        # locations of a "dense" format of shape [num_nodes, size].
        nodes, perm = nodes.sort()
        neighbors, e_id = neighbors[perm], e_id[perm]

        n_id = nodes.unique()
        self.__assoc__[n_id] = torch.arange(n_id.numel(), device=n_id.device)

        dense_id = torch.arange(nodes.size(0), device=nodes.device) % self.k
        dense_id += self.__assoc__[nodes].mul_(self.k)

        dense_e_id = e_id.new_full((n_id.numel() * self.k, ), -1)
        dense_e_id[dense_id] = e_id
        dense_e_id = dense_e_id.view(-1, self.k)

        dense_neighbors = e_id.new_empty(n_id.numel() * self.k)
        dense_neighbors[dense_id] = neighbors
        dense_neighbors = dense_neighbors.view(-1, self.k)

        # Collect new and old interactions...
        e_id = torch.cat([self.e_id[n_id, :self.k], dense_e_id], dim=-1)
        neighbors = torch.cat(
            [self.neighbors[n_id, :self.k], dense_neighbors], dim=-1)

        # And sort them based on `e_id`.
        e_id, perm = e_id.topk(self.k, dim=-1)
        self.e_id[n_id] = e_id
        self.neighbors[n_id] = torch.gather(neighbors, 1, perm)
    
    def reset(self):
        self.cur_e_id = 0
        self.e_id.fill_(-1)

    def attach_last_update(self,last_t):
        self.last_update = last_t

    def sync(self,sampler):
        self.cur_e_id  = sampler.cur_e_id
        self.neighbors = copy.deepcopy(sampler.neighbors)
        self.e_id      = copy.deepcopy(sampler.e_id)
        self.__assoc__ = copy.deepcopy(sampler.__assoc__)

    
class FastTemporalEdgeCollator(EdgeCollator):
    def _collate_with_negative_sampling(self,items):
        items = _prepare_tensor(self.g_sampling,items,'items',False)
        # Here node id will not change
        pair_graph = self.g.edge_subgraph(items,preserve_nodes = True)
        induced_edges = pair_graph.edata[dgl.EID]

        neg_srcdst_raw = self.negative_sampler(self.g,items)
        neg_srcdst = {self.g.canonical_etypes[0]: neg_srcdst_raw}
        dtype = list(neg_srcdst.values())[0][0].dtype
        neg_edges = {
            etype: neg_srcdst.get(etype, (torch.tensor([], dtype=dtype), torch.tensor([], dtype=dtype)))
            for etype in self.g.canonical_etypes}

        neg_pair_graph = dgl.heterograph(
            neg_edges, {ntype: self.g.number_of_nodes(ntype) for ntype in self.g.ntypes})
        pair_graph, neg_pair_graph = dgl.transform.compact_graphs([pair_graph,neg_pair_graph])
        # Need to remap id

        pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][pair_graph.ndata[dgl.NID]]
        neg_pair_graph.ndata[dgl.NID] = self.g.ndata[dgl.NID][neg_pair_graph.ndata[dgl.NID]]

        pair_graph.edata[dgl.EID] = induced_edges
        
        seed_nodes = pair_graph.ndata[dgl.NID]
        blocks = self.block_sampler.sample_blocks(self.g_sampling,seed_nodes)
        blocks[0].ndata['timestamp'] = torch.zeros(blocks[0].num_nodes()).double()
        input_nodes = blocks[0].edges()[1]
        
        # update sampler 
        _src = self.g.ndata[dgl.NID][self.g.edges()[0][items]]
        _dst = self.g.ndata[dgl.NID][self.g.edges()[1][items]]
        self.block_sampler.add_edges(_src,_dst)
        return input_nodes,pair_graph,neg_pair_graph,blocks
        
    def collator(self,items):
        result = super().collate(items)
        # Copy the feature from parent graph
        _pop_subgraph_storage(result[1],self.g)
        _pop_subgraph_storage(result[2],self.g)
        _pop_blocks_storage(result[-1],self.g_sampling)
        return result
        