
# TODO: Define the TGN logic and model
from numpy.core.fromnumeric import mean
import torch
import torch.nn as nn
import dgl
import dgl.function as fn
from modules import MemoryModule, MemoryOperation, TemporalGATConv, LinkPredictor


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

        print(self.edge_feat_dim,self.memory_dim,self.temporal_dim,self.embedding_dim,self.num_heads)
        self.embedding_attn      = TemporalGATConv(self.edge_feat_dim,
                                                   self.memory_dim,
                                                   self.temporal_dim,
                                                   self.embedding_dim,
                                                   self.num_heads)

        self.linkpredictor = LinkPredictor(embedding_dim)

    # Each batch forward once
    def forward(self,srcs,dsts,negs,timestamps,subg,mode):
        # TODO: Attention for embedding using src, dst, neg edge flow while loop
        pred_list_pos = []
        pred_list_neg = []
        for src,dst,neg,ts in zip(srcs,dsts,negs,timestamps):
            # Does negative sampling means allow overlap
            pos_subg = self.edge_sampler(src,dst,ts,mode)
            pos_subg = dgl.remove_self_loop(pos_subg)
            pos_subg = dgl.add_self_loop(pos_subg)
            n_ts = ts.repeat(pos_subg.num_nodes())
            nodes = [src,dst]
            memory_cell = self.memory.memory[nodes,:]
            print(pos_subg)
            z_pos = self.embedding_attn(pos_subg,memory_cell,n_ts) #[src_embedding,dst_embedding]
            neg_subg = self.node_sampler(neg,ts,mode)
            n_ts = ts.repeat(neg_subg.num_nodes())
            memory_cell = self.memory.memory[neg,:]
            z_neg = self.embedding_attn(neg_subg,memory_cell,n_ts)
            pred_list_pos.append(self.linkpredictor(z_pos[0],z_pos[1]))
            pred_list_neg.append(self.linkpredictor(z_pos[0],z_neg.squeeze()))
        # TODO: Update memory module.
        subg = dgl.remove_self_loop(subg)
        subg = dgl.add_self_loop(subg)
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID],new_g.ndata['s'])
        pred_pos = torch.cat(pred_list_pos)
        pred_neg = torch.cat(pred_list_neg)
        return pred_pos, pred_neg

    def attach_sampler(self,edge_temporal_sampler,node_temporal_sampler):
        # Temporal sampler decoupled from main code allow future change
        self.edge_sampler = edge_temporal_sampler
        self.node_sampler = node_temporal_sampler

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()
