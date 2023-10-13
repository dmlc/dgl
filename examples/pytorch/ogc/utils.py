import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def symmetric_normalize_adjacency(graph):
    """ Symmetric normalize graph adjacency matrix. """
    adj = graph.adjacency_matrix()
    in_degs = graph.in_degrees().float()
    in_norm = torch.pow(in_degs, -0.5).unsqueeze(-1)
    degi = torch.diag(torch.squeeze(torch.t(in_norm)))
    degi = sp.coo_matrix(degi).tocsr()
    adj = sp.csr_matrix((adj.val.cpu(), (adj.row.cpu(), adj.col.cpu())), shape=adj.shape)
    adj = degi.dot(adj.dot(degi))
    return adj


class LinearNeuralNetwork(nn.Module):
    def __init__(self, nfeat, nclass, bias=True):
        super(LinearNeuralNetwork, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=bias)

    def forward(self, x):
        return self.W(x)

    def test(self, U, g):
        self.eval()
        with torch.no_grad():
            output = self(U)
            pred = output.argmax(dim=-1)
            labels = g.ndata["label"]
            test_mask = g.ndata["test_mask"]
            tv_mask = g.ndata["train_mask"] + g.ndata["val_mask"]
            loss_tv = F.mse_loss(output[tv_mask],
                                 F.one_hot(labels).float()[tv_mask])
            accs = []
            for mask in [tv_mask, test_mask]:
                accs.append(
                    float((pred[mask] == labels[mask]).sum()/mask.sum()))
        return loss_tv.item(), accs[0], accs[1], pred

    def update_W(self, U, g, eta_W):
        optimizer = optim.SGD(self.parameters(), lr=eta_W)
        self.train()
        optimizer.zero_grad()
        output = self(U)
        labels = g.ndata["label"]
        tv_mask = g.ndata["train_mask"] + g.ndata["val_mask"]
        loss_tv = F.mse_loss(output[tv_mask],
                             F.one_hot(labels).float()[tv_mask],
                             reduction='sum')
        loss_tv.backward()
        optimizer.step()
        return self(U).data, self.W.weight.data
