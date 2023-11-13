import dgl.sparse as dglsp
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNeuralNetwork(nn.Module):
    def __init__(self, nfeat, nclass, bias=True):
        super(LinearNeuralNetwork, self).__init__()
        self.W = nn.Linear(nfeat, nclass, bias=bias)

    def forward(self, x):
        return self.W(x)


def symmetric_normalize_adjacency(graph):
    """Symmetric normalize graph adjacency matrix."""
    indices = torch.stack(graph.edges())
    n = graph.num_nodes()
    adj = dglsp.spmatrix(indices, shape=(n, n))
    deg_invsqrt = dglsp.diag(adj.sum(0)) ** -0.5
    return deg_invsqrt @ adj @ deg_invsqrt


def model_test(model, embeds):
    model.eval()
    with torch.no_grad():
        output = model(embeds)
        pred = output.argmax(dim=-1)
        test_mask, tv_mask = model.test_mask, model.tv_mask
        loss_tv = F.mse_loss(output[tv_mask], model.label_one_hot[tv_mask])
    accs = []
    for mask in [tv_mask, test_mask]:
        accs.append(float((pred[mask] == model.label[mask]).sum() / mask.sum()))
    return loss_tv.item(), accs[0], accs[1], pred
