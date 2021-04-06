import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.base import DGLError
from dgl.ops import edge_softmax
import dgl.function as fn


class MergeLayer(nn.Module):
    """Merge two tensor into one
    Which is useful as skip connection in Merge GAT's input with output

    Parameter
    ----------
    dim1 : int
        dimension of first input tensor

    dim2 : int
        dimension of second input tensor

    dim3 : int
        hidden dimension after first merging

    dim4 : int
        output dimension

    Example
    ----------
    >>> merger = MergeLayer(10,10,10,5)
    >>> input1 = torch.ones(4,10)
    >>> input2 = torch.ones(4,10)
    >>> merger(input1,input2)
    tensor([[-0.1578,  0.1842,  0.2373,  1.2656,  1.0362],
        [-0.1578,  0.1842,  0.2373,  1.2656,  1.0362],
        [-0.1578,  0.1842,  0.2373,  1.2656,  1.0362],
        [-0.1578,  0.1842,  0.2373,  1.2656,  1.0362]],
       grad_fn=<AddmmBackward>)
    """

    def __init__(self, dim1, dim2, dim3, dim4):
        super(MergeLayer, self).__init__()
        self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
        self.fc2 = torch.nn.Linear(dim3, dim4)
        self.act = torch.nn.ReLU()

        torch.nn.init.xavier_normal_(self.fc1.weight)
        torch.nn.init.xavier_normal_(self.fc2.weight)

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        h = self.act(self.fc1(x))
        return self.fc2(h)


class MsgLinkPredictor(nn.Module):
    """Predict Pair wise link from pos subg and neg subg
    use message passing.

    Use Two layer MLP on edge to predict the link probability

    Parameters
    ----------
    embed_dim : int
        dimension of each each feature's embedding

    Example
    ----------
    >>> linkpred = MsgLinkPredictor(10)
    >>> pos_g = dgl.graph(([0,1,2,3,4],[1,2,3,4,0]))
    >>> neg_g = dgl.graph(([0,1,2,3,4],[2,1,4,3,0]))
    >>> x = torch.ones(5,10)
    >>> linkpred(x,pos_g,neg_g)
    (tensor([[0.0902],
         [0.0902],
         [0.0902],
         [0.0902],
         [0.0902]], grad_fn=<AddmmBackward>),
    tensor([[0.0902],
         [0.0902],
         [0.0902],
         [0.0902],
         [0.0902]], grad_fn=<AddmmBackward>))
    """

    def __init__(self, emb_dim):
        super(MsgLinkPredictor, self).__init__()
        self.src_fc = nn.Linear(emb_dim, emb_dim)
        self.dst_fc = nn.Linear(emb_dim, emb_dim)
        self.out_fc = nn.Linear(emb_dim, 1)

    def link_pred(self, edges):
        src_hid = self.src_fc(edges.src['embedding'])
        dst_hid = self.dst_fc(edges.dst['embedding'])
        score = F.relu(src_hid+dst_hid)
        score = self.out_fc(score)
        return {'score': score}

    def forward(self, x, pos_g, neg_g):
        # Local Scope?
        pos_g.ndata['embedding'] = x
        neg_g.ndata['embedding'] = x

        pos_g.apply_edges(self.link_pred)
        neg_g.apply_edges(self.link_pred)

        pos_escore = pos_g.edata['score']
        neg_escore = neg_g.edata['score']
        return pos_escore, neg_escore


class TimeEncode(nn.Module):
    """Use finite fourier series with different phase and frequency to encode
    time different between two event

    ..math::
        \Phi(t) = [\cos(\omega_0t+\psi_0),\cos(\omega_1t+\psi_1),...,\cos(\omega_nt+\psi_n)] 

    Parameter
    ----------
    dimension : int
        Length of the fourier series. The longer it is , 
        the more timescale information it can capture

    Example
    ----------
    >>> tecd = TimeEncode(10)
    >>> t = torch.tensor([[1]])
    >>> tecd(t)
    tensor([[[0.5403, 0.9950, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
          1.0000, 1.0000]]], dtype=torch.float64, grad_fn=<CosBackward>)
    """

    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .double().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).double())

    def forward(self, t):
        t = t.unsqueeze(dim=2)
        output = torch.cos(self.w(t))
        return output


class MemoryModule(nn.Module):
    """Memory module as well as update interface

    The memory module stores both historical representation in last_update_t

    Parameters
    ----------
    n_node : int
        number of node of the entire graph

    hidden_dim : int
        dimension of memory of each node

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py;
                     examples/pytorch/tgn/train.py 

    """

    def __init__(self, n_node, hidden_dim):
        super(MemoryModule, self).__init__()
        self.n_node = n_node
        self.hidden_dim = hidden_dim
        self.reset_memory()

    def reset_memory(self):
        self.last_update_t = nn.Parameter(torch.zeros(
            self.n_node).float(), requires_grad=False)
        self.memory = nn.Parameter(torch.zeros(
            (self.n_node, self.hidden_dim)).float(), requires_grad=False)

    def backup_memory(self):
        """
        Return a deep copy of memory state and last_update_t
        For test new node, since new node need to use memory upto validation set
        After validation, memory need to be backed up before run test set without new node
        so finally, we can use backup memory to update the new node test set
        """
        return self.memory.clone(), self.last_update_t.clone()

    def restore_memory(self, memory_backup):
        """Restore the memory from validation set

        Parameters
        ----------
        memory_backup : (memory,last_update_t)
            restore memory based on input tuple
        """
        self.memory = memory_backup[0].clone()
        self.last_update_t = memory_backup[1].clone()

    # Which is used for attach to subgraph
    def get_memory(self, node_idxs):
        return self.memory[node_idxs, :]

    # When the memory need to be updated
    def set_memory(self, node_idxs, values):
        self.memory[node_idxs, :] = values

    def set_last_update_t(self, node_idxs, values):
        self.last_update_t[node_idxs] = values

    # For safety check
    def get_last_update(self, node_idxs):
        return self.last_update_t[node_idxs]

    def detach_memory(self):
        """
        Disconnect the memory from computation graph to prevent gradient be propagated multiple
        times
        """
        self.memory.detach_()


class MemoryOperation(nn.Module):
    """ Memory update using message passing manner, update memory based on positive
    pair graph of each batch with recurrent module GRU or RNN

    Message function
    ..math::
        m_i(t) = concat(memory_i(t^-),TimeEncode(t),v_i(t))

    v_i is node feature at current time stamp

    Aggregation function
    ..math::
        \bar{m}_i(t) = last(m_i(t_1),...,m_i(t_b))

    Update function
    ..math::
        memory_i(t) = GRU(\bar{m}_i(t),memory_i(t-1))

    Parameters
    ----------

    updater_type : str
        indicator string to specify updater

        'rnn' : use Vanilla RNN as updater

        'gru' : use GRU as updater

    memory : MemoryModule
        memory content for update

    e_feat_dim : int
        dimension of edge feature

    temporal_dim : int
        length of fourier series for time encoding

    Example
    ----------
    Please refers to examples/pytorch/tgn/tgn.py
    """

    def __init__(self, updater_type, memory, e_feat_dim, temporal_dim):
        super(MemoryOperation, self).__init__()
        updater_dict = {'gru': nn.GRUCell, 'rnn': nn.RNNCell}
        self.memory = memory
        memory_dim = self.memory.hidden_dim
        self.message_dim = memory_dim+memory_dim+e_feat_dim+temporal_dim
        self.updater = updater_dict[updater_type](input_size=self.message_dim,
                                                  hidden_size=memory_dim)
        self.memory = memory
        self.temporal_encoder = TimeEncode(temporal_dim)

    # Here assume g is a subgraph from each iteration
    def stick_feat_to_graph(self, g):
        # How can I ensure order of the node ID
        g.ndata['timestamp'] = self.memory.last_update_t[g.ndata[dgl.NID]]
        g.ndata['memory'] = self.memory.memory[g.ndata[dgl.NID]]

    def msg_fn_cat(self, edges):
        src_delta_time = edges.data['timestamp'] - edges.src['timestamp']
        time_encode = self.temporal_encoder(src_delta_time.unsqueeze(
            dim=1)).view(len(edges.data['timestamp']), -1)
        ret = torch.cat([edges.src['memory'], edges.dst['memory'],
                         edges.data['feats'], time_encode], dim=1)
        return {'message': ret, 'timestamp': edges.data['timestamp']}

    def agg_last(self, nodes):
        timestamp, latest_idx = torch.max(nodes.mailbox['timestamp'], dim=1)
        ret = nodes.mailbox['message'].gather(1, latest_idx.repeat(
            self.message_dim).view(-1, 1, self.message_dim)).view(-1, self.message_dim)
        return {'message_bar': ret.reshape(-1, self.message_dim), 'timestamp': timestamp}

    def update_memory(self, nodes):
        # It should pass the feature through RNN
        ret = self.updater(
            nodes.data['message_bar'].float(), nodes.data['memory'].float())
        return {'memory': ret}

    def forward(self, g):
        self.stick_feat_to_graph(g)
        g.update_all(self.msg_fn_cat, self.agg_last, self.update_memory)
        return g


class TemporalGATConv(nn.Module):
    """Dot Product based embedding with temporal encoding
    it will each node will compute the attention weight with other
    nodes using other node's memory as well as edge features

    Aggregation:
    ..math::
        h_i^{(l+1)} = ReLu(\sum_{j\in \mathcal{N}(i)} \alpha_{i, j} (h_j^{(l)}(t)||e_{jj}||TimeEncode(t-t_j)))

        \alpha_{i, j} = \mathrm{softmax_i}(\frac{QK^T}{\sqrt{d_k}})V

    K,Q,V computation:
    ..math::
        K = W_k[memory_{src}(t),memory_{dst}(t),TimeEncode(t_{dst}-t_{src})]
        Q = W_q[memory_{src}(t),memory_{dst}(t),TimeEncode(t_{dst}-t_{src})]
        V = W_v[memory_{src}(t),memory_{dst}(t),TimeEncode(t_{dst}-t_{src})]

    Parameters
    ----------

    edge_feats : int
        dimension of edge feats

    memory_feats : int
        dimension of memory feats

    temporal_feats : int
        length of fourier series of time encoding

    num_heads : int
        number of head in multihead attention

    allow_zero_in_degree : bool
        Whether allow some node have indegree == 0 to prevent silence evaluation

    Example
    ----------
    >>> attn = TemporalGATConv(2,2,5,2,2,False)
    >>> star_graph = dgl.graph(([1,2,3,4,5],[0,0,0,0,0]))
    >>> star_graph.edata['feats'] = torch.ones(5,2).double()
    >>> star_graph.edata['timestamp'] = torch.zeros(5).double()
    >>> memory = torch.ones(6,2)
    >>> ts = torch.random.rand(6).double()
    >>> star_graph = dgl.add_self_loop(star_graph)
    >>> attn(graph,memory,ts)
    tensor([[-0.0924, -0.3842],
        [-0.0840, -0.3539],
        [-0.0842, -0.3543],
        [-0.0838, -0.3536],
        [-0.0856, -0.3568],
        [-0.0858, -0.3572]], grad_fn=<AddmmBackward>)
    """

    def __init__(self,
                 edge_feats,
                 memory_feats,
                 temporal_feats,
                 out_feats,
                 num_heads,
                 allow_zero_in_degree=False):
        super(TemporalGATConv, self).__init__()
        self._edge_feats = edge_feats
        self._memory_feats = memory_feats
        self._temporal_feats = temporal_feats
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._num_heads = num_heads

        self.fc_Q = nn.Linear(self._memory_feats+self._temporal_feats,
                              self._out_feats*self._num_heads, bias=False)
        self.fc_K = nn.Linear(self._memory_feats+self._edge_feats +
                              self._temporal_feats, self._out_feats*self._num_heads, bias=False)
        self.fc_V = nn.Linear(self._memory_feats+self._edge_feats +
                              self._temporal_feats, self._out_feats*self._num_heads, bias=False)
        self.merge = MergeLayer(
            self._memory_feats, self._out_feats*self._num_heads, 512, self._out_feats)
        self.temporal_encoder = TimeEncode(self._temporal_feats)

    def weight_fn(self, edges):
        t0 = torch.zeros_like(edges.dst['timestamp'])
        q = torch.cat([edges.dst['s'],
                       self.temporal_encoder(t0.unsqueeze(dim=1)).view(len(t0), -1)], dim=1)
        time_diff = edges.data['timestamp']-edges.src['timestamp']
        time_encode = self.temporal_encoder(
            time_diff.unsqueeze(dim=1)).view(len(t0), -1)
        k = torch.cat(
            [edges.src['s'], edges.data['feats'], time_encode], dim=1)
        squeezed_k = self.fc_K(
            k.float()).view(-1, self._num_heads, self._out_feats)
        squeezed_q = self.fc_Q(
            q.float()).view(-1, self._num_heads, self._out_feats)
        ret = torch.sum(squeezed_q*squeezed_k, dim=2)
        return {'a': ret, 'efeat': squeezed_k}

    def msg_fn(self, edges):
        ret = edges.data['sa'].view(-1, self._num_heads, 1)*edges.data['efeat']
        return {'attn': ret}

    def forward(self, graph, memory, ts):
        graph = graph.local_var()  # Using local scope for graph
        if not self._allow_zero_in_degree:
            if(graph.in_degrees() == 0).any():
                raise DGLError('There are 0-in-degree nodes in the graph, '
                               'output for those nodes will be invalid. '
                               'This is harmful for some applications, '
                               'causing silent performance regression. '
                               'Adding self-loop on the input graph by '
                               'calling `g = dgl.add_self_loop(g)` will resolve '
                               'the issue. Setting ``allow_zero_in_degree`` '
                               'to be `True` when constructing this module will '
                               'suppress the check and let the code run.')

        #print("Shape: ",memory.shape,ts.shape)
        graph.srcdata.update({'s': memory, 'timestamp': ts})
        graph.dstdata.update({'s': memory, 'timestamp': ts})

        # Dot product Calculate the attentio weight
        graph.apply_edges(self.weight_fn)

        # Edge softmax
        graph.edata['sa'] = edge_softmax(
            graph, graph.edata['a'])/(self._out_feats**0.5)

        # Update dst node Here msg_fn include edge feature
        graph.update_all(self.msg_fn, fn.sum('attn', 'agg_u'))

        rst = graph.dstdata['agg_u']
        # Implement skip connection
        rst = self.merge(rst.view(-1, self._num_heads *
                                  self._out_feats), graph.dstdata['s'])
        return rst
