from networkx.classes.graph import Graph
import numpy as np
import scipy.sparse as sparse
import torch
import torch.nn as nn
import dgl
import dgl.nn as dglnn
from dgl.base import DGLError
import dgl.function as fn
from dgl.nn.functional import edge_softmax

class DiffConv(nn.Module):
    in_graph_list = []
    out_graph_list = []
    init = True
    def __init__(self,in_feats,out_feats,k,dir='both'):
        super(DiffConv,self).__init__()
        self.in_feats = in_feats
        self.out_feats= out_feats
        self.k = k
        self.dir = dir
        self.num_graphs = self.k-1 if self.dir == 'both' else 2*self.k-2
        self.project_fcs = nn.ModuleList()
        for i in range(self.num_graphs):
            self.project_fcs.append(nn.Linear(self.in_feats,self.out_feats,bias=False))
        self.merger = nn.Parameter(torch.randn(self.num_graphs+1))

    @staticmethod
    def attach_graph(g,k): # g need preprocess for weight adjustment
        device  = g.device
        # Idempotent
        DiffConv.out_graph_list = []
        DiffConv.in_graph_list = []
        wadj,ind,outd = DiffConv.get_weight_matrix(g)
        adj = sparse.coo_matrix(wadj/outd.cpu().numpy())
        outg = dgl.from_scipy(adj,eweight_name='weight').to(device)
        outg.edata['weight'] = outg.edata['weight'].float().to(device)
        DiffConv.out_graph_list.append(outg)
        for i in range(k-1):
            DiffConv.out_graph_list.append(DiffConv.diffuse(DiffConv.out_graph_list[-1],wadj,outd))
        adj = sparse.coo_matrix(wadj.T/ind.cpu().numpy())
        ing = dgl.from_scipy(adj,eweight_name='weight').to(device)
        ing.edata['weight'] = ing.edata['weight'].float().to(device)
        DiffConv.in_graph_list.append(ing)
        for i in range(k-1):
            DiffConv.in_graph_list.append(DiffConv.diffuse(DiffConv.in_graph_list[-1],wadj.T,ind))

    @staticmethod
    def get_weight_matrix(g):
        adj = g.adj(scipy_fmt='coo')
        ind = g.in_degrees()
        outd = g.out_degrees()
        weight = g.edata['weight']
        adj.data = weight.cpu().numpy()
        return adj,ind,outd

    @staticmethod
    def diffuse(progress_g,weighted_adj,degree):
        device = progress_g.device
        progress_adj = progress_g.adj(scipy_fmt='coo')
        progress_adj.data = progress_g.edata['weight'].cpu().numpy()
        ret_adj = sparse.coo_matrix(progress_adj@(weighted_adj/degree.cpu().numpy()))
        ret_graph = dgl.from_scipy(ret_adj,eweight_name='weight').to(device)
        ret_graph.edata['weight'] = ret_graph.edata['weight'].float().to(device)
        return ret_graph

    def forward(self,g,x):
        if DiffConv.init == True:
            DiffConv.attach_graph(g,self.k)
            DiffConv.init = False
        if g.num_nodes()!=DiffConv.in_graph_list[0].num_nodes():
            DiffConv.attach_graph(g,self.k)
        feat_list = [] # Each element in the feature list should be [N,out_feats]
        if self.dir == 'both':
            graph_list = DiffConv.in_graph_list+DiffConv.out_graph_list
        elif self.dir == 'in':
            graph_list = DiffConv.in_graph_list
        elif self.dir == 'out':
            graph_list = DiffConv.out_graph_list

        for i in range(self.num_graphs):
            g = graph_list[i]
            with g.local_scope():
                g.ndata['n'] = self.project_fcs[i](x)
                #g.apply_edges(fn.copy_u('n','e'))
                g.update_all(fn.u_mul_e('n','weight','e'),fn.sum('e','feat'))
                feat_list.append(g.ndata['feat'])
                # Each feat has shape [N,q_feats]
        feat_list.append(self.project_fcs[-1](x))
        feat_list = torch.cat(feat_list).view(len(feat_list),-1,self.out_feats)
        ret = (self.merger*feat_list.permute(1,2,0)).permute(2,0,1).mean(0)
        return ret

# Basically it is Equivalent to GAT with a head
# How should we handle the edge feature
class WeightedGATConv(dglnn.GATConv):
    def forward(self,graph,feat,get_attention=False):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            if isinstance(feat, tuple):
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(-1, self._num_heads, self._out_feats)
                else:
                    feat_src = self.fc_src(h_src).view(-1, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(-1, self._num_heads, self._out_feats)
            else:
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    -1, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
            # NOTE: GAT paper uses "first concatenation then linear projection"
            # to compute attention scores, while ours is "first projection then
            # addition", the two approaches are mathematically equivalent:
            # We decompose the weight vector a mentioned in the paper into
            # [a_l || a_r], then
            # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
            # Our implementation is much efficient because we do not need to
            # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
            # addition could be optimized with DGL's built-in function u_add_v,
            # which further speeds up computation and saves memory footprint.
            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            # compute edge attention, el and er are a_l Wh_i and a_r Wh_j respectively.
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            e = self.leaky_relu(graph.edata.pop('e'))
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
            # compute weighted attention
            graph.edata['a'] = (graph.edata['a'].permute(1,2,0)*graph.edata['weight']).permute(2,0,1)
            # message passing
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            rst = graph.dstdata['ft']
            # residual
            if self.res_fc is not None:
                resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
                rst = rst + resval
            # activation
            if self.activation:
                rst = self.activation(rst)

            if get_attention:
                return rst, graph.edata['a']
            else:
                return rst

class GatedGAT(nn.Module):
    def __init__(self,in_feats,out_feats,map_feats,num_heads):
        super(GatedGAT,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.map_feats = map_feats
        self.num_heads = num_heads
        self.gatlayer = WeightedGATConv(self.in_feats,
                                      self.out_feats,
                                      self.num_heads)
        self.gate_fn = nn.Linear(2*self.in_feats+self.map_feats,self.num_heads)
        self.gate_m = nn.Linear(self.in_feats,self.map_feats)
        self.merger_layer = nn.Linear(self.in_feats+self.out_feats,self.out_feats)

    def agg_fn(self,nodes):
        max_z,_ = (self.gate_m(nodes.mailbox['z'])).max(1)
        mean_z = torch.mean(nodes.mailbox['z'],dim=1)
        debug = torch.cat([nodes.data['x'],max_z,mean_z],dim=1)
        ret = self.gate_fn(torch.cat([nodes.data['x'],max_z,mean_z],dim=1))
        ret = torch.sigmoid(ret)
        return {'agg':ret}

    # The edge weight need to be considered when aggregate
    def forward(self,g,x):
        # It need to each node will have individual GAT Conv 
        with g.local_scope():
            g.ndata['x'] = x
            g.update_all(fn.copy_u('x','z'),self.agg_fn) # [num_node,num_head,out_feats]
            gate = g.ndata['agg']
            attn_out = self.gatlayer(g,x) # [num_node,num_head,out_feat]
            node_num = g.num_nodes()
            gated_out = ((gate.view(-1)*attn_out.view(-1,self.out_feats).T).T).view(node_num,self.num_heads,self.out_feats)
            gated_out = gated_out.mean(1)
            merge = self.merger_layer(torch.cat([x,gated_out],dim=1))
            return merge

# Special Handling is needed for DiffConv
# Here net should be partially initialized
class GraphGRUCell(nn.Module):
    def __init__(self,in_feats,out_feats,net):
        super(GraphGRUCell,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.dir = dir
        # net can be any GNN model
        self.r_net = net(in_feats+out_feats,out_feats)
        self.u_net = net(in_feats+out_feats,out_feats)
        self.c_net = net(in_feats+out_feats,out_feats)
        # Manually add bias Bias should be the same for each node and added to each node
        self.r_bias = nn.Parameter(torch.rand(out_feats))
        self.u_bias = nn.Parameter(torch.rand(out_feats))
        self.c_bias = nn.Parameter(torch.rand(out_feats)) 


    def forward(self,g,x,h):
        r = torch.sigmoid(self.r_net(g,torch.cat([x,h],dim=1)) + self.r_bias)
        u = torch.sigmoid(self.u_net(g,torch.cat([x,h],dim=1)) + self.u_bias)
        h_ = r*h
        c = torch.sigmoid(self.c_net(g,torch.cat([x,h_],dim=1)) + self.c_bias)
        new_h = u*h + (1-u)*c
        return new_h

class StackedEncoder(nn.Module):
    def __init__(self,in_feats,out_feats,num_layers,net):
        super(StackedEncoder,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_layers = num_layers
        self.net = net
        self.layers = nn.ModuleList()
        if self.num_layers<=0:
            raise DGLError("Layer Number must be greater than 0! ")
        self.layers.append(GraphGRUCell(self.in_feats,self.out_feats,self.net))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(self.out_feats,self.out_feats,self.net))

    # hidden_states should be a list which for different layer
    def forward(self,g,x,hidden_states):
        hiddens = []
        for i,layer in enumerate(self.layers):
            x = layer(g,x,hidden_states[i])
            hiddens.append(x)
        return x, hiddens

class StackedDecoder(nn.Module):
    '''Here decoder means decoding the 
    '''
    def __init__(self,in_feats,hid_feats,out_feats,num_layers,net):
        super(StackedDecoder,self).__init__()
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.out_feats= out_feats
        self.num_layers = num_layers
        self.net = net
        self.out_layer = nn.Linear(self.hid_feats,self.out_feats)
        self.layers = nn.ModuleList()
        if self.num_layers <= 0:
            raise DGLError("Layer Number must be greater than 0!")
        self.layers.append(GraphGRUCell(self.in_feats,self.hid_feats,net))
        for _ in range(self.num_layers-1):
            self.layers.append(GraphGRUCell(self.hid_feats,self.hid_feats,net))

    def forward(self,g,x,hidden_states):
        hiddens = []
        for i,layer in enumerate(self.layers):
            x = layer(g,x,hidden_states[i])
            hiddens.append(x)
        x = self.out_layer(x)
        return x,hiddens

class DCRNN(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 seq_len,
                 num_layers,
                 net,
                 decay_steps):
        super(DCRNN,self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.net = net
        self.decay_steps = decay_steps
        
        # The task need to be done is 
        self.encoder = StackedEncoder(self.in_feats,
                                       self.out_feats,
                                       self.num_layers,
                                       self.net)

        # Out put need to be able to feed in as input
        self.decoder = StackedDecoder(self.in_feats,
                                       self.out_feats,
                                       self.in_feats,
                                       self.num_layers,
                                       self.net)
    # Threshold For Teacher Forcing
    def compute_thresh(self,batch_cnt):
        return self.decay_steps/(self.decay_steps + np.exp(batch_cnt / self.decay_steps))

    def encode(self,g,inputs,device):
        hidden_states = [torch.zeros(g.num_nodes(),self.out_feats).to(device) for _ in range(self.num_layers)]
        for i in range(self.seq_len):
            _,hidden_states = self.encoder(g,inputs[i],hidden_states)

        return hidden_states

    def decode(self,g,teacher_states,hidden_states,batch_cnt,device):
        outputs = []
        inputs = torch.zeros(g.num_nodes(),self.in_feats).to(device)
        for i in range(self.seq_len):
            if np.random.random() < self.compute_thresh(batch_cnt) and self.training:
                inputs,hidden_states = self.decoder(g,teacher_states[i],hidden_states)
            else:
                inputs,hidden_states = self.decoder(g,inputs,hidden_states)
            outputs.append(inputs)
        outputs = torch.stack(outputs) # Skeptical about shape
        return outputs

    def forward(self,g,inputs,teacher_states,batch_cnt,device):
        hidden = self.encode(g,inputs,device)
        outputs = self.decode(g,teacher_states,hidden,batch_cnt,device)
        return outputs
        
# need to define the Unit test of the model
if __name__ == "__main__":
    from functools import partial
    # Unit test for diffconv
    diffconv = DiffConv(in_feats=10,out_feats = 5,k=3)
    # Assume edge have weight
    g = dgl.graph(([0,1,2,3,4,5],[1,2,3,4,5,0]))
    # Should be 1D tensor
    g.edata['weight'] = torch.ones(6).float()
    x = torch.rand(6,10)
    out = diffconv(g,x)
    assert(out.shape==torch.Size([6,5]))

    # Unit test for GaAN
    ggat = GatedGAT(10,5,6,8)
    out = ggat(g,x)
    assert(out.shape==torch.Size([6,5]))

    # Unit Test for GRU Cell

    ggat = partial(GatedGAT,map_feats=10,num_heads=8)
    diffconv = partial(DiffConv,k=3)
    fake_hidden = torch.ones(6,5)
    graph_gru_cell = GraphGRUCell(10,5,ggat)
    out = graph_gru_cell(g,x,fake_hidden)
    assert(out.shape == torch.Size([6,5]))
    graph_gru_cell = GraphGRUCell(10,5,diffconv)
    out = graph_gru_cell(g,x,fake_hidden)
    assert(out.shape == torch.Size([6,5]))

    # Unit test for sequence encoder
    ggat_encoder = StackedEncoder(10,5,3,ggat)
    fake_hidden = [torch.ones(6,5),torch.ones(6,5),torch.ones(6,5)]
    out,hid = ggat_encoder(g,x,fake_hidden)
    assert(out.shape==torch.Size([6,5]))
    assert(len(hid)==3)
    assert(hid[0].shape==torch.Size([6,5]))
    assert(hid[1].shape==torch.Size([6,5]))
    assert(hid[2].shape==torch.Size([6,5]))

    # Unit test for sequence decoder
    ggat_decoder = StackedDecoder(10,5,2,3,ggat)
    out,hid = ggat_decoder(g,x,fake_hidden)
    assert(out.shape==torch.Size([6,2]))
    assert(len(hid)==3)
    assert(hid[0].shape==torch.Size([6,5]))
    assert(hid[1].shape==torch.Size([6,5]))
    assert(hid[2].shape==torch.Size([6,5]))
    
    # Unit test for DCRNN

    dcrnn = DCRNN(10,5,3,3,ggat,1000)
    inputs = torch.ones(2,3,6,10)
    teacher_states = torch.ones(3,6,10)
    outputs = dcrnn(g,inputs,teacher_states,1)
    assert(outputs.shape==torch.Size([3,6,10]))
