import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as FN
from .. import randomwalk
from ..utils import cuda

def create_embeddings(n_nodes, n_features):
    return nn.Parameter(torch.randn(n_nodes, n_features))

def mix_embeddings(h, ndata, emb, proj):
    '''Combine node-specific trainable embedding ``h`` with categorical inputs
    (projected by ``emb``) and numeric inputs (projected by ``proj``).
    '''
    e = []
    for key, value in ndata.items():
        if value.dtype == torch.int64:
            e.append(emb[key](cuda(value)))
        elif value.dtype == torch.float32:
            e.append(proj[key](cuda(value)))
    return cuda(h) + torch.stack(e, 0).sum(0)

def get_embeddings(h, nodeset):
    return h[nodeset]

def put_embeddings(h, nodeset, new_embeddings):
    n_nodes = nodeset.shape[0]
    n_features = h.shape[1]
    return h.scatter(0, nodeset[:, None].expand(n_nodes, n_features), new_embeddings)

def safediv(a, b):
    b = torch.where(b == 0, torch.ones_like(b), b)
    return a / b

def init_weight(w, func_name, nonlinearity):
    getattr(nn.init, func_name)(w, gain=nn.init.calculate_gain(nonlinearity))

def init_bias(w):
    nn.init.constant_(w, 0)

class PinSageConv(nn.Module):
    def __init__(self, in_features, out_features, hidden_features):
        super(PinSageConv, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features

        self.Q = nn.Linear(in_features, hidden_features)
        self.W = nn.Linear(in_features + hidden_features, out_features)

        init_weight(self.Q.weight, 'xavier_uniform_', 'leaky_relu')
        init_weight(self.W.weight, 'xavier_uniform_', 'leaky_relu')
        init_bias(self.Q.bias)
        init_bias(self.W.bias)

    message = [FN.src_mul_edge('h', 'ppr_weight', 'h_w'),
               FN.copy_src('ppr_weight', 'w')]
    reduce = [FN.sum('h_w', 'h_agg'), FN.sum('w', 'w')]

    def apply(self, nodes):
        h_agg = safediv(nodes['h_agg'], nodes['w'])
        h = nodes['h']
        h_concat = torch.cat([h, h_agg], 1)
        h_new = F.leaky_relu(self.W(h_concat))
        h_new = safediv(h_new, h_new.norm(dim=1, keepdim=True))
        return {'h': h_new}


class PinSage(nn.Module):
    def __init__(self, feature_sizes, use_feature=False, G=None):
        super(PinSage, self).__init__()
        self.in_features = feature_sizes[0]
        self.out_features = feature_sizes[-1]
        self.n_layers = len(feature_sizes) - 1

        self.convs = nn.ModuleList()
        for i in range(self.n_layers):
            self.convs.append(PinSageConv(
                feature_sizes[i], feature_sizes[i+1], feature_sizes[i+1]))

        self.use_feature = use_feature

        if use_feature:
            self.emb = nn.ModuleDict()
            self.proj = nn.ModuleDict()

            for key, scheme in G.node_attr_schemes().items():
                if scheme.dtype == torch.int64:
                    self.emb[key] = nn.Embedding(
                            G.ndata[key].max().item() + 1,
                            self.in_features,
                            padding_idx=0)
                elif scheme.dtype == torch.float32:
                    self.proj[key] = nn.Sequential(
                            nn.Linear(scheme.shape[0], self.in_features),
                            nn.LeakyReLU(),
                            )

    def forward(self, nf, h):
        '''
        nf: NodeFlow.  Features are already copied from the original graph.
        '''
        if self.use_feature:
            nf.layers[0].data['h'] = mix_embeddings(
                    h, nf.layers[0].data, self.emb, self.proj)

        for i in range(self.n_layers):
            nf.block_compute(
                    i,
                    self.convs[i].message,
                    self.convs[i].reduce,
                    self.convs[i].apply)

        return nf.layers[-1].data['h']
