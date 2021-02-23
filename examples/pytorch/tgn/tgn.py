import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from modules import MemoryModule, MemoryOperation, TemporalGATConv, LinkPredictor
import time
from pyinstrument import Profiler

# The TGN Model will take two different types of subgraphs as input
# Any thing related to dataloader should not appear here
class TGN(nn.Module):
    def __init__(self,
                 edge_feat_dim,
                 memory_dim,
                 temporal_dim,
                 embedding_dim,
                 num_heads,
                 num_nodes, # entire graph
                 n_neighbors = 10,
                 memory_updater_type='gru'):
        super(TGN,self).__init__()
        self.memory_dim          = memory_dim
        self.edge_feat_dim       = edge_feat_dim
        self.temporal_dim        = temporal_dim
        self.embedding_dim       = embedding_dim
        self.num_heads           = num_heads
        self.n_neighbors         = n_neighbors
        self.memory_updater_type = memory_updater_type
        self.num_nodes           = num_nodes

        self.memory              = MemoryModule(self.num_nodes,
                                                self.memory_dim)

        self.memory_ops          = MemoryOperation(self.memory_updater_type,
                                                   self.memory,
                                                   self.edge_feat_dim,
                                                   self.temporal_dim)

        #print(self.edge_feat_dim,self.memory_dim,self.temporal_dim,self.embedding_dim,self.num_heads)
        self.embedding_attn      = TemporalGATConv(self.edge_feat_dim,
                                                   self.memory_dim,
                                                   self.temporal_dim,
                                                   self.embedding_dim,
                                                   self.num_heads)

        self.linkpredictor = LinkPredictor(embedding_dim)
        self.profiler = Profiler()
    
    # Need to sample subgraph batchs
    # The returned result also need to match id
    # Assume new_pos and new_neg are lists
    def temporal_batch(self,srcs,dsts,negs,timestamps,mode):
        # Each positive subgraph contains 2 real mattered nodes
        batch_graphs = []
        batch_memorys = []
        batch_ts = []
        batch_offset = 0
        pos_src_id = []
        pos_dst_id = []
        neg_id = []
        for src,dst,neg,ts in zip(srcs,dsts,negs,timestamps):
            # Here new pos and new neg specify the node in subgraph
            # After merge the new_pos will be relative position
            # Need a scrolling offset
            new_pos,pos_subg = self.sampler([src,dst],ts,mode)
            pos_src_id.append(new_pos[0]+batch_offset)
            pos_dst_id.append(new_pos[1]+batch_offset)
            batch_graphs.append(pos_subg)
            batch_memorys.append(self.memory.memory[pos_subg.ndata[dgl.NID],:])
            batch_ts.append(ts.repeat(pos_subg.num_nodes()))
            batch_offset += pos_subg.num_nodes()

            new_neg,neg_subg = self.sampler(neg,ts,mode)
            neg_id.append(new_neg[0]+batch_offset)
            batch_graphs.append(neg_subg)
            batch_memorys.append(self.memory.memory[neg_subg.ndata[dgl.NID],:])
            batch_ts.append(ts.repeat(neg_subg.num_nodes()))
            batch_offset += neg_subg.num_nodes()
        batch_graph = dgl.batch(batch_graphs)
        batch_memory= torch.cat(batch_memorys,dim=0)
        batch_t = torch.cat(batch_ts,dim=0)
        pos_src_id = torch.tensor(pos_src_id)
        pos_dst_id = torch.tensor(pos_dst_id)
        neg_id = torch.tensor(neg_id)
        return batch_graph,batch_memory,batch_t,pos_src_id,pos_dst_id,neg_id

    def fast_sample(self,srcs,dsts,negs,timestamps,mode):
        # Generate unique nodes for batching
        nodes_unique = torch.cat([srcs,dsts,negs]).unique().tolist()
        timestamps = self.memory.last_update_t[nodes_unique]
        nodes_unique_sub,emb_graph = self.sampler(nodes_unique,timestamps,mode)
        
        emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID],:]
        emb_t      = self.memory.last_update_t[emb_graph.ndata[dgl.NID]].double()
        # Map the srcs, dsts, negs to id in subg.
        # use multiple dictionary since the size of each list is not equal
        work2sub_dict= dict(zip(nodes_unique,nodes_unique_sub))
        
        # Remap might contain duplication
        for i,j,k in zip(srcs,dsts,negs):
            pos_src_id = work2sub_dict[int(i)]
            pos_dst_id = work2sub_dict[int(j)]
            neg_id     = work2sub_dict[int(k)]
        return emb_graph,emb_memory,emb_t,pos_src_id,pos_dst_id,neg_id

    def batch_forward(self,srcs,dsts,negs,timestamps,subg,mode):
        pred_pos,pred_neg = self.embed(srcs,dsts,negs,timestamps,mode)
        self.update_memory(subg)
        return pred_pos,pred_neg

    def fast_embed(self,srcs,dsts,negs,timestamps,mode):
        # Invoking graph sampler for the embedding subgraph
        # Here the pos_src_id and pos_dst_id as well as neg_id might involve duplication
        #self.profiler.start()
        emb_graph,emb_memory,emb_t,pos_src_id,pos_dst_id,neg_id = self.fast_sample(srcs,dsts,negs,timestamps,mode)
        embedding = self.embedding_attn(emb_graph,emb_memory,emb_t)
        pred_pos = self.linkpredictor(embedding[pos_src_id],embedding[pos_dst_id])
        pred_neg = self.linkpredictor(embedding[pos_src_id],embedding[neg_id])
        #self.profiler.stop()
        #print(self.profiler.output_text(unicode=True,color=True))
        return pred_pos,pred_neg

    def embed(self,srcs,dsts,negs,timestamps,mode):
        batch_graph,batch_memory,batch_t,pos_src_id,pos_dst_id,neg_id = self.temporal_batch(srcs,dsts,negs,timestamps,mode)
        embedding = self.embedding_attn(batch_graph,batch_memory,batch_t)
        pred_pos = self.linkpredictor(embedding[pos_src_id],embedding[pos_dst_id])
        pred_neg = self.linkpredictor(embedding[pos_src_id],embedding[neg_id])
        return pred_pos, pred_neg

    def update_memory(self,subg):
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID],new_g.ndata['memory'])
        self.memory.set_last_update_t(new_g.ndata[dgl.NID],new_g.ndata['timestamp'])

    def attach_sampler(self,temporal_sampler):
        self.sampler = temporal_sampler

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()
