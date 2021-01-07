import torch as th
import torch.nn as nn
import dgl.function as fn

from layers import drop_node, MLP

def GRANDConv(graph, feats, order):
    '''
    Parameters
    -----------
    graph: dgl.Graph
        The input graph
    feats: Tensor (n_nodes * feat_dim)
        Node features
    order: int 
        Propagation Steps
    '''
    with graph.local_scope():
        
        ''' Calculate Symmetric normalized adjacency matrix   \hat{A} '''
        degs = graph.in_degrees().float().clamp(min=1)
        norm = th.pow(degs, -0.5).to(feats.device).unsqueeze(1)

        graph.ndata['norm'] = norm
        graph.apply_edges(fn.u_mul_v('norm', 'norm', 'weight')) 
        
        ''' Graph Conv '''
        x = feats
        y = 0+feats

        for i in range(order):
            graph.ndata['h'] = x
            graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'h'))
            x = graph.ndata.pop('h')
            y.add_(x)

    return y /(order + 1)

class GRAND(nn.Module):
    r"""

    Parameters
    -----------
    in_dim: int
        Input feature size. i.e, the number of dimensions of: math: `H^{(i)}`.
    hid_dim: int
        Hidden feature size.
    n_class: int
        Number of classes.
    S: int
        Number of Augmentation samples
    K: int
        Number of Propagation Steps
    node_dropout: float
        Dropout rate on node features.
    input_dropout: float
        Dropout rate of the input layer of a MLP
    hidden_dropout: float
        Dropout rate of the hidden layer of a MLPx
    batchnorm: bool, optional
        If True, use batch normalization.

    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 n_class,
                 S = 1,
                 K = 3,
                 node_dropout=0.0,
                 input_droprate = 0.0, 
                 hidden_droprate = 0.0,
                 batchnorm=False):

        super(GRAND, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.S = S
        self.K = K
        self.n_class = n_class

        self.mlp = MLP(in_dim, hid_dim, n_class, input_droprate, hidden_droprate, batchnorm)
        
        self.dropout = node_dropout
        self.node_dropout = nn.Dropout(node_dropout)

    def forward(self, graph, feats, training = True):
        
        X = feats
        S = self.S
        
        if training: # Training Mode
            output_list = []
            for s in range(S):
                drop_feat = drop_node(X, self.dropout, True)  # Drop node
                feat = GRANDConv(graph, drop_feat, self.K)    # Graph Convolution
                output_list.append(th.log_softmax(self.mlp(feat), dim=-1))  # Prediction
        
            return output_list
        else:   # Inference Mode
            drop_feat = drop_node(X, self.dropout, False) 
            X =  GRANDConv(graph, drop_feat, self.K)

            return th.log_softmax(self.mlp(X), dim = -1)
