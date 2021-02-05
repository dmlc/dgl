
# TODO: Define the TGN logic and model
from numpy.core.fromnumeric import mean
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

    # Each batch forward once
    # Need to accelerate the code here
    def forward(self,srcs,dsts,negs,timestamps,subg,mode):
        # TODO: Set neighbors' ts to be zero to prevent mistake.embedding[new_pos[0]]
        pred_list_pos = []
        pred_list_neg = []
        last_t = time.time()
        self.profiler.start()
        # Need to enable batch training and test the speed up
        for src,dst,neg,ts in zip(srcs,dsts,negs,timestamps):
            # Real link embedding and compute score
            new_pos,pos_subg = self.edge_sampler([src,dst],ts,mode)
            n_ts = ts.repeat(pos_subg.num_nodes())
            nodes = pos_subg.ndata[dgl.NID]
            memory_cell = self.memory.memory[nodes,:]
            embedding = self.embedding_attn(pos_subg,memory_cell,n_ts)
            z_pos = [embedding[new_pos[0]],embedding[new_pos[1]]]

            # Fake link embedding and compute score
            new_neg,neg_subg = self.node_sampler(neg,ts,mode)
            n_ts = ts.repeat(neg_subg.num_nodes())
            nodes = neg_subg.ndata[dgl.NID]
            memory_cell = self.memory.memory[nodes,:].view(len(nodes),-1)
            z_neg = self.embedding_attn(neg_subg,memory_cell,n_ts)[new_neg]
            # Print shape
            

            # Link prediction via MLP
            pred_list_pos.append(self.linkpredictor(z_pos[0],z_pos[1]))
            pred_list_neg.append(self.linkpredictor(z_pos[0],z_neg[0]))

        # Update memory module.
        print("Embedding: ",time.time()-last_t)
        last_t = time.time()
        subg = dgl.remove_self_loop(subg)
        subg = dgl.add_self_loop(subg)
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID],new_g.ndata['s'])
        pred_pos = torch.cat(pred_list_pos)
        pred_neg = torch.cat(pred_list_neg)
        print("Memory update: ",time.time()-last_t)
        self.profiler.stop()
        print(self.profiler.output_text(unicode=True,color=True))
        return pred_pos, pred_neg
    
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
            new_pos,pos_subg = self.edge_sampler([src,dst],ts,mode)
            pos_src_id.append(new_pos[0]+batch_offset)
            pos_dst_id.append(new_pos[1]+batch_offset)
            batch_graphs.append(pos_subg)
            batch_memorys.append(self.memory.memory[pos_subg.ndata[dgl.NID],:])
            batch_ts.append(ts.repeat(pos_subg.num_nodes()))
            batch_offset += pos_subg.num_nodes()

            new_neg,neg_subg = self.node_sampler(neg,ts,mode)
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

    def batch_forward(self,srcs,dsts,negs,timestamps,subg,mode):
        batch_graph,batch_memory,batch_t,pos_src_id,pos_dst_id,neg_id = self.temporal_batch(srcs,dsts,negs,timestamps,mode)
        embedding = self.embedding_attn(batch_graph,batch_memory,batch_t)
        pred_pos = self.linkpredictor(embedding[pos_src_id],embedding[pos_dst_id])
        pred_neg = self.linkpredictor(embedding[pos_src_id],embedding[neg_id])
        subg = dgl.remove_self_loop(subg)
        subg = dgl.add_self_loop(subg)
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID],new_g.ndata['s'])
        return pred_pos,pred_neg

    def embed(self,srcs,dsts,negs,timestamps,mode):
        batch_graph,batch_memory,batch_t,pos_src_id,pos_dst_id,neg_id = self.temporal_batch(srcs,dsts,negs,timestamps,mode)
        embedding = self.embedding_attn(batch_graph,batch_memory,batch_t)
        pred_pos = self.linkpredictor(embedding[pos_src_id],embedding[pos_dst_id])
        pred_neg = self.linkpredictor(embedding[pos_src_id],embedding[neg_id])
        return pred_pos, pred_neg

    def update_memory(self,subg):
        subg = dgl.remove_self_loop(subg)
        subg = dgl.add_self_loop(subg)
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID],new_g.ndata['s'])

    def attach_sampler(self,edge_temporal_sampler,node_temporal_sampler):
        # Temporal sampler decoupled from main code allow future change
        self.edge_sampler = edge_temporal_sampler
        self.node_sampler = node_temporal_sampler

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()
