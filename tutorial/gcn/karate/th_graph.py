import numpy as np
from matplotlib import pyplot as plt

import networkx as nx
import torch

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

edges = np.loadtxt("../edges.txt", dtype=np.int32)
nodes = np.loadtxt("../nodes.txt", dtype=np.int32)
G = nx.Graph()

for i in range(4):
    G.add_nodes_from(nodes[nodes[:, 1] == i][:, 0], labels=i)
G.add_edges_from(edges)

nodes = G.nodes(data=True)
values = []
for i in range(1, G.number_of_nodes() + 1):
    values.append(nodes[i]['labels'])
edge_index = torch.from_numpy(np.array(G.edges())).long().t()
y = torch.from_numpy(np.array(values))
# x = torch.eye(G.number_of_nodes()).float()
x = torch.zeros((34, 2)).float()

train_mask = torch.zeros(G.number_of_nodes())
train_mask.data[0] = 1
train_mask.data[2] = 1
train_mask.data[8] = 1
train_mask.data[4] = 1

val_mask = 1 - train_mask
data = Data(x=x, edge_index=edge_index - 1, y=y)
data.train_mask = train_mask.to(torch.uint8).to(device)
data.val_mask = val_mask.to(torch.uint8).to(device)


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(data.num_features, 16)
        self.conv2 = GCNConv(16, data.num_classes)

    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        return F.log_softmax(x, dim=1)


model, data = Net().to(device), data.to(device).contiguous()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

EPOCH = 10

for i in range(EPOCH):
    output = model(data)
    optimizer.zero_grad()

    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    # Validation
    logits = model()
    mask = data.val_mask
    pred = logits[mask].max(1)[1]
    acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
    print(f"Validation Accuracy: {acc}")
