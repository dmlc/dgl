import copy
import torch.nn as nn
import dgl
from modules import MemoryModule, MemoryOperation, MsgLinkPredictor, TemporalTransformerConv, TimeEncode


class TGN(nn.Module):
    def __init__(self,
                 edge_feat_dim,
                 memory_dim,
                 temporal_dim,
                 embedding_dim,
                 num_heads,
                 num_nodes,
                 n_neighbors=10,
                 memory_updater_type='gru',
                 layers=1):
        super(TGN, self).__init__()
        self.memory_dim = memory_dim
        self.edge_feat_dim = edge_feat_dim
        self.temporal_dim = temporal_dim
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.n_neighbors = n_neighbors
        self.memory_updater_type = memory_updater_type
        self.num_nodes = num_nodes
        self.layers = layers

        self.temporal_encoder = TimeEncode(self.temporal_dim)

        self.memory = MemoryModule(self.num_nodes,
                                   self.memory_dim)

        self.memory_ops = MemoryOperation(self.memory_updater_type,
                                          self.memory,
                                          self.edge_feat_dim,
                                          self.temporal_encoder)

        self.embedding_attn = TemporalTransformerConv(self.edge_feat_dim,
                                                      self.memory_dim,
                                                      self.temporal_encoder,
                                                      self.embedding_dim,
                                                      self.num_heads,
                                                      layers=self.layers,
                                                      allow_zero_in_degree=True)

        self.msg_linkpredictor = MsgLinkPredictor(embedding_dim)

    def embed(self, postive_graph, negative_graph, blocks):
        emb_graph = blocks[0]
        emb_memory = self.memory.memory[emb_graph.ndata[dgl.NID], :]
        emb_t = emb_graph.ndata['timestamp']
        embedding = self.embedding_attn(emb_graph, emb_memory, emb_t)
        emb2pred = dict(
            zip(emb_graph.ndata[dgl.NID].tolist(), emb_graph.nodes().tolist()))
        # Since postive graph and negative graph has same is mapping
        feat_id = [emb2pred[int(n)] for n in postive_graph.ndata[dgl.NID]]
        feat = embedding[feat_id]
        pred_pos, pred_neg = self.msg_linkpredictor(
            feat, postive_graph, negative_graph)
        return pred_pos, pred_neg

    def update_memory(self, subg):
        new_g = self.memory_ops(subg)
        self.memory.set_memory(new_g.ndata[dgl.NID], new_g.ndata['memory'])
        self.memory.set_last_update_t(
            new_g.ndata[dgl.NID], new_g.ndata['timestamp'])

    # Some memory operation wrappers
    def detach_memory(self):
        self.memory.detach_memory()

    def reset_memory(self):
        self.memory.reset_memory()

    def store_memory(self):
        memory_checkpoint = {}
        memory_checkpoint['memory'] = copy.deepcopy(self.memory.memory)
        memory_checkpoint['last_t'] = copy.deepcopy(self.memory.last_update_t)
        return memory_checkpoint

    def restore_memory(self, memory_checkpoint):
        self.memory.memory = memory_checkpoint['memory']
        self.memory.last_update_time = memory_checkpoint['last_t']
