# TODO: Use implement a test scripts
import dgl
import torch
import torch.nn as nn
from dgl.ops import edge_softmax
import dgl.function as fn
import numpy as np
from dgl.base import DGLError

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4):
    super(MergeLayer,self).__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)

class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

class TemporalGATConv(nn.Module):
    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_feats,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False):
        super(TemporalGATConv,self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats  = memory_feats
        self._temporal_feats= temporal_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads
        

        self.fc_Q = nn.Linear(self._memory_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.fc_K = nn.Linear(self._memory_feats+self._edge_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.fc_V = nn.Linear(self._memory_feats+self._edge_feats+self._temporal_feats,self._out_feats*self._num_heads,bias=False)
        self.merge= MergeLayer(self._memory_feats,self._out_feats*self._num_heads,512,self._out_feats)
        self.temporal_encoder = TimeEncode(self._temporal_feats)

    # TODO: Check dimension which might be problematic
    def weight_fn(self,edges): # need to know the size of temporal feature.
        t0 = torch.zeros_like(edges.dst['timestamp'])
        q = torch.cat([edges.dst['s'],self.temporal_encoder(t0.unsqueeze(dim=1)).view(len(t0),-1)],dim=1)
        time_diff = edges.data['timestamp']-edges.src['timestamp']
        k = torch.cat([edges.src['s'],edges.data['feats'],self.temporal_encoder(time_diff.unsqueeze(dim=1)).view(len(t0),-1)],dim=1)
        squeezed_k = self.fc_K(k).view(-1,self._num_heads,self._out_feats)
        squeezed_q = self.fc_Q(q).view(-1,self._num_heads,self._out_feats)
        ret = torch.sum(squeezed_q*squeezed_k,dim=2)
        return {'a':ret,'efeat':squeezed_k} 

    def msg_fn(self,edges):
        ret = edges.data['sa'].view(-1,self._num_heads,1)*edges.data['efeat']
        return {'attn':ret}

    # Need to assume the feature include edge feature.
    # Here the input graph is the temporal subgraph
    # Assume feat is memory
    def forward(self,graph,memory,ts):
        graph = graph.local_var() # Using local scope for graph
        if not self._allow_zero_in_degree:
            if(graph.in_degrees()==0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

            
        graph.srcdata.update({'s':memory,'timestamp':ts})
        graph.dstdata.update({'s':memory,'timestamp':ts})

        # Dot product Calculate the attentio weight
        graph.apply_edges(self.weight_fn)

        # Edge softmax
        graph.edata['sa'] = edge_softmax(graph,graph.edata['a'])

        # Update dst node
        graph.update_all(self.msg_fn,fn.sum('attn','agg_u'))

        rst = graph.dstdata['agg_u']
        # Implement skip connection
        rst = self.merge(rst.view(-1,self._num_heads*self._out_feats),graph.dstdata['s'])
        return rst


# Test Code here

# Use a toy graph for correctness testing
g = dgl.graph(([0,0,0,0,1,1,1,1],[2,3,4,5,2,4,6,7]))

memory = torch.ones((g.num_nodes(),500),dtype=torch.float32)
timestamps = torch.tensor([9,8,7,5,5,4,3,2]).float()

g.edata['feats']     = torch.ones((g.num_edges(),172)).float()
g.edata['timestamp'] = torch.tensor([3,3,3,3,3,3,3,3]).float()

model = TemporalGATConv(172,500,20,10,2,allow_zero_in_degree=True)

print(model(g,memory,timestamps).shape)
