import sys
sys.path.append('..')
import dgl
import dgl.function as fn
from dgl import NID, EID
from dgl import edge_subgraph
from dgl.data import MovieLensDataset
from dgl.nn.pytorch import GraphConv, HeteroGraphConv
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.nn.functional import softmax
import numpy as np

'''
ml-100k, with node features, gcmc
rmse: 0.9473 (original implementation of gcmc example: 0.9448)
'''

class BiDecoder(nn.Module):
    # codes adopted from GCMC example
    def __init__(self, in_units, num_classes, num_basis=2, dropout_rate=0.0):
        super(BiDecoder, self).__init__()
        self._num_basis = num_basis
        self.dropout = nn.Dropout(dropout_rate)
        self.Ps = nn.ModuleList(
            nn.Linear(in_units, in_units, bias=False) for _ in range(num_basis)
        )
        self.combine_basis = nn.Linear(self._num_basis, num_classes, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, graph, ufeat, ifeat):
        with graph.local_scope():
            ufeat = self.dropout(ufeat)
            ifeat = self.dropout(ifeat)
            graph.nodes["movie"].data["h"] = ifeat
            graph.nodes["user"].data["h"] = ufeat
            basis_out = []
            for i in range(self._num_basis):
                self.Ps[i](graph.nodes["user"].data["h"])
                graph.apply_edges(fn.u_dot_v("h", "h", "sr"), etype="user-movie")
                basis_out.append(graph.edges['user-movie'].data["sr"])
            out = th.cat(basis_out, dim=1)
            out = self.combine_basis(out)
        return out
    
class RateConv(nn.Module):
    def __init__(self, rate_list, in_dim, out_dim):
        super(RateConv, self).__init__()
        self.rate_list = rate_list
        self.convs = nn.ModuleList()
        for _ in rate_list:
            self.convs.append(GraphConv(in_dim, out_dim))

    def forward(self, g, feat):
        with g.local_scope():
            etype = g.etypes[0].split('-')
            g.ndata['h'] = {etype[0]: feat[0], etype[1]: feat[1]}
            h_list = []
            for rate, conv in zip(self.rate_list, self.convs):
                edges = (g.edata['rate'] == rate).nonzero().flatten()
                _g = edge_subgraph(g, edges)
                h = conv(_g, (_g.nodes[etype[0]].data['h'], _g.nodes[etype[1]].data['h']))
                dst_h = th.zeros(feat[1].shape[0], h.shape[1]).to(h.device)
                dst_h[_g.nodes[etype[1]].data[NID]] = h
                h_list.append(dst_h)
            h = th.concat(h_list, dim=1)
        return h

class Model(nn.Module):
    def __init__(self, rate_list, in_dim:dict, hid_dim:int):
        super(Model, self).__init__()
        self.conv = HeteroGraphConv({
            'user-movie': RateConv(rate_list, in_dim['user'], hid_dim),
            'movie-user': RateConv(rate_list, in_dim['movie'], hid_dim),
        })
        self.decoder = BiDecoder(hid_dim*len(rate_list), len(rate_list))
    def forward(self, g, feat):
        h = self.conv(g, feat)
        h = self.decoder(g, h['user'], h['movie'])
        return h
    
def convert_labels(labels, rate_list):
    return th.tensor([rate_list.index(l) for l in labels]).to('cuda')

def convert_preds(logits, rate_list):
    return (softmax(logits, dim=1).detach().cpu() * th.tensor(rate_list)).sum(dim=1)

hid_dim, out_dim, epochs, patience, lr = 500, 75, 2000, 100, 0.01
dataset = MovieLensDataset('ml-100k', valid_ratio=0.2, force_reload=True)
graph = dataset[0]
in_dim = {'user': graph.nodes['user'].data['feat'].shape[1], 'movie': graph.nodes['movie'].data['feat'].shape[1]}
rate = graph.edges['user-movie'].data['rate']
rate_list = rate.unique(sorted=True).tolist()
model = Model(rate_list, in_dim, hid_dim)
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()

train_mask, valid_mask, test_mask =\
    graph.edges['user-movie'].data['train_mask'],\
    graph.edges['user-movie'].data['valid_mask'],\
    graph.edges['user-movie'].data['test_mask']
labels = convert_labels(rate, rate_list)

best_valid_rmse = np.inf
best_test_rmse = None
best_epoch = None
cnt = 0

model = model.to('cuda')
graph = graph.to('cuda')
print(f"Start training")
model.train()
for _ in range(epochs):
    optimizer.zero_grad()
    logits = model(graph, graph.ndata['feat'])
    loss = loss_fn(logits[train_mask], labels[train_mask])
    loss.backward()
    optimizer.step()

    preds = convert_preds(logits, rate_list)
    train_rmse = th.sqrt(((preds[train_mask].float() - rate[train_mask].float()) ** 2).mean()).item()
    valid_rmse = th.sqrt(((preds[valid_mask].float() - rate[valid_mask].float()) ** 2).mean()).item()
    test_rmse = th.sqrt(((preds[test_mask].float() - rate[test_mask].float()) ** 2).mean()).item()

    if valid_rmse < best_valid_rmse:
        best_valid_rmse = valid_rmse
        best_test_rmse = test_rmse
        best_epoch = _
        cnt = 0
    else:
        cnt += 1
    if best_test_rmse < 0.8:
        continue
    if cnt == patience:
        break
    print(f'epoch: {_}/{epochs}, best epoch: {best_epoch} | patience: {cnt}/{patience} | loss: {loss.item():.4f}')
    print(f'rmse: train {train_rmse:.4f}, valid {valid_rmse:.4f}, test {test_rmse:.4f}')
    print(f'best_valid_rmse: {best_valid_rmse:.4f}, best_test_rmse: {best_test_rmse:.4f}')
    print()

print(f"Finish")







