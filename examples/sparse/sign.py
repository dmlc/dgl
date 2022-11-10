import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from dgl.data import CoraGraphDataset
from dgl.mock_sparse import create_from_coo, identity, diag

def sign(A, X, num_hops):
    X_sign = [X]
    for _ in range(num_hops):
        X = A @ X
        X_sign.append(X)
    return X_sign

class Model(nn.Module):
    def __init__(self, in_size, num_hops, num_classes, hidden_size=256):
        super().__init__()

        num_feats = num_hops + 1
        self.linear = nn.ModuleList([
            nn.Linear(in_size, hidden_size) for _ in range(num_feats)
        ])
        self.pred = nn.Linear(hidden_size * num_feats, num_classes)

    def forward(self, X):
        X = torch.cat([
            self.linear[i](X[i]) for i in range(len(X))
        ], dim=1)
        return self.pred(F.relu(X))

def evaluate(X, Y, model):
    model.eval()
    with torch.no_grad():
        logits = model(X)
        _, Y_hat = torch.max(logits, dim=1)
        return (Y_hat == Y).float().mean().item()

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data = CoraGraphDataset()
g = data[0].to(dev)
src, dst = g.edges()
N = g.num_nodes()
A = create_from_coo(dst, src, shape=(N, N))

I = identity(A.shape, device=dev)
A_hat = A + I
D_hat = diag(A.sum(dim=1)) ** -0.5
A_hat = D_hat @ A_hat @ D_hat

X = g.ndata["feat"]
Y = g.ndata["label"]
train_mask = g.ndata["train_mask"]
test_mask = g.ndata["test_mask"]

num_hops = 2
X_sign = sign(A_hat, X, num_hops)
X_sign_train, X_sign_test = [], []
for feat in X_sign:
    X_sign_train.append(feat[train_mask])
    X_sign_test.append(feat[test_mask])
Y_train = Y[train_mask]
Y_test = Y[test_mask]

model = Model(X.shape[1], num_hops, data.num_classes).to(dev)
loss_func = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=3e-3)

model.train()
for epoch in range(10):
    pred = model(X_sign_train)
    loss = loss_func(pred, Y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    train_acc = evaluate(X_sign_train, Y_train, model)
    print(f'epoch {epoch} | train acc {train_acc}')

test_acc = evaluate(X_sign_test, Y_test, model)
print(f'test acc {test_acc}')
