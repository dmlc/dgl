import dgl
import dgl.data
import dgl.mock_sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy
import tqdm

dataset = dgl.data.CoraGraphDataset()

graph = dataset[0]
# The paper created the hypergraph by "each time one vertex in the
# graph is selected as the centroid and its connected vertices
# are used to generate one hyperedge including the centroid
# itself".  In this case, the incidence matrix of the hypergraph
# is the same as the adjacency matrix of the original graph (with
# self-loops).
# We follow the paper assuming that the rows of the incidence matrix
# are nodes and the columns are edges.

# QUESTION: can we either (1) easily convert a PyTorch sparse matrix
# to a DGL sparse matrix, or (2) make g.adj() a DGL sparse matrix
# in the future?
adj = dgl.add_self_loop(graph).adj().coalesce()
row, col = adj.indices()
values = adj.values()
H = dgl.mock_sparse.create_from_coo(row, col, values)

X = graph.ndata["feat"]
Y = graph.ndata["label"]
train_mask = graph.ndata["train_mask"]
val_mask = graph.ndata["val_mask"]
test_mask = graph.ndata["test_mask"]


class HGNN(nn.Module):
    def __init__(self, in_size, out_size, hidden_dims=16):
        super().__init__()

        self.Theta1 = nn.Linear(in_size, hidden_dims)
        self.Theta2 = nn.Linear(hidden_dims, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, H, X):
        d_V = H.sum(1)  # node degree
        d_E = H.sum(0)  # edge degree
        D_V_invsqrt = dgl.mock_sparse.diag(d_V ** -0.5)  # D_V ** (-1/2)
        D_E_inv = dgl.mock_sparse.diag(d_E ** -1)  # D_E ** (-1)
        # Minor suggestion: could we have dgl.mock_sparse.eye(N)?
        W = dgl.mock_sparse.diag(torch.ones(d_E.shape[0]))

        conv = D_V_invsqrt @ H @ W @ D_E_inv @ H.T @ D_V_invsqrt

        X = conv @ self.Theta1(self.dropout(X))
        X = F.relu(X)
        X = conv @ self.Theta2(self.dropout(X))
        return X


model = HGNN(X.shape[1], dataset.num_classes)
opt = torch.optim.Adam(model.parameters(), lr=0.001)
best_val_acc, best_test_acc = 0, 0

with tqdm.trange(500) as tq:
    for epoch in tq:
        # Train
        model.train()
        Y_hat = model(H, X)
        loss = F.cross_entropy(Y_hat[train_mask], Y[train_mask])
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Evaluate
        model.eval()
        Y_hat = model(H, X)
        val_acc = accuracy(Y_hat[val_mask], Y[val_mask])
        test_acc = accuracy(Y_hat[test_mask], Y[test_mask])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        tq.set_postfix(
            {
                "Loss": f"{loss.item():.5f}",
                "Val acc": f"{val_acc:.5f}",
                "Test acc": f"{test_acc:.5f}",
            },
            refresh=False,
        )
print(f"Best val acc: {best_val_acc} Best test acc: {best_test_acc}")
