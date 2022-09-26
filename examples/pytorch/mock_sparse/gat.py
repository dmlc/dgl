import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl import AddSelfLoop
import argparse

from dgl.mock_sparse import create_from_coo, diag, identity

class GATConv(nn.Module):
    def __init__(self, in_size, out_size, n_heads):
        super(GATConv, self).__init__()
        self.W = nn.Parameter(torch.Tensor(in_size, out_size * n_heads))
        self.a_l = nn.Parameter(torch.Tensor(1, n_heads, out_size))
        self.a_r = nn.Parameter(torch.Tensor(1, n_heads, out_size))

    def forward(self, A, h):
    	Wh = (h @ self.W).view(-1, self.n_heads, self.out_size) # |V| x N_h x D_o
    	Wh1 = (Wh * self.a_l).sum(2)  # |V| x N_h
    	Wh2 = (Wh * self.a_r).sum(2)  # |V| x N_h
        # option.1: use SDDMM operator; faster but may not be intuitive
        # e = dgl.sparse.sddmm(A, Wh1, Wh2, op='+')  # nonzeros with vector data
        # e = dgl.sparse.sddmm_add(A, Wh1, Wh2)  #..
        # # Or, option.2: use lookup; syntax is closer to Wh_i and Wh_j but more costly
        # e = Wh1[A.i] + Wh2[A.j]  # |E| x N_h

        # e = nn.LeakyReLU(e)
        # A_hat = dgl.sparse.softmax(A(e), dim=1)  # sparse softmax; normalize non-zero entries
        # h_prime = A_hat @ Wh

    	return torch.relu(h)

class GAT(nn.Module):
    def __init__(self, in_size, out_size, n_heads):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_size, out_size, n_heads))
        # self.dropout = nn.Dropout(0.5)

    def forward(self, A, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(A, h)
        return h

def evaluate(A, features, labels, mask, model):
    model.eval()
    with torch.no_grad():
        logits = model(A, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


def train(A, features, labels, masks, model):
    # define train/val samples, loss function and optimizer
    train_mask = masks[0]
    val_mask = masks[1]
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(A, features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(A, features, labels, val_mask, model)
        print("Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} "
              . format(epoch, loss.item(), acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset name ('cora', 'citeseer', 'pubmed').")
    parser.add_argument("--n_heads", type=int, default=1,
                        help="number of attention heads.")
    args = parser.parse_args()
    print(f'Training with DGL SparseMatrix GATConv module.')

    # load and preprocess dataset
    transform = AddSelfLoop()  # by default, it will first remove self-loops to prevent duplication
    if args.dataset == 'cora':
        data = CoraGraphDataset(transform=transform)
    elif args.dataset == 'citeseer':
        data = CiteseerGraphDataset(transform=transform)
    elif args.dataset == 'pubmed':
        data = PubmedGraphDataset(transform=transform)
    else:
        raise ValueError('Unknown dataset: {}'.format(args.dataset))
    g = data[0]
    n_heads = args.n_heads
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    g = g.int().to(device)
    features = g.ndata['feat']
    labels = g.ndata['label']
    masks = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

    row, col = g.adj_sparse('coo')
    A = create_from_coo(row, col, shape=(g.number_of_nodes(), g.number_of_nodes()))

    # create GAT model
    in_size = features.shape[1]
    out_size = data.num_classes
    model = GAT(in_size, out_size, n_heads).to(device)

    # # model training
    # print('Training...')
    # train(A, features, labels, masks, model)

    # # test the model
    # print('Testing...')
    # acc = evaluate(A, features, labels, masks[2], model)
    # print("Test accuracy {:.4f}".format(acc))
