
import torch
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborSampler
import dgl.graphbolt as gb
import torch.nn.functional as F

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  
            if i == 0:
                x = self.conv1((x, x_target), edge_index)
                x = F.relu(x)
            else:
                x = self.conv2((x, x_target), edge_index)
                x = F.relu(x)
            if i != len(adjs) - 1:
                x = F.dropout(x, p=0.5, training=self.training)
        return x


def load_data():
    dataset = gb.BuiltinDataset("ogbn-arxiv").load()
    graph = dataset.graph
    
    features = dataset.feature

    features_tensor = features.read('node', None, 'feat')
    
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set

    train_labels = train_set._items[1]  
    valid_labels = valid_set._items[1]
    test_labels = test_set._items[1]


    graph = dataset.graph

    num_classes = dataset.tasks[0].metadata["num_classes"]

    csc_indptr = graph.csc_indptr
    rows = torch.repeat_interleave(torch.arange(csc_indptr.size(0) - 1, device=csc_indptr.device), csc_indptr[1:] - csc_indptr[:-1])
    indices = graph.indices
    cols = indices
    edge_index =  torch.stack([rows, cols], dim=0)
    

    pyg_data = Data(x=features_tensor, edge_index=edge_index, y=train_labels)  
    

    num_nodes = csc_indptr.size(0) - 1

    return pyg_data, train_labels, valid_labels, test_labels, train_set, valid_set, test_set, num_nodes
   

def create_data_loaders(pyg_data, train_labels, valid_labels, test_labels, train_set, valid_set, test_set):
    train_mask = train_set._items[0]
    valid_mask = valid_set._items[0]
    test_mask = test_set._items[0]
    train_loader = NeighborSampler(pyg_data.edge_index, node_idx=train_mask, sizes=[10, 10], batch_size=32, shuffle=True, num_workers=4)
    valid_loader = NeighborSampler(pyg_data.edge_index, node_idx=valid_mask, sizes=[10, 10], batch_size=32, shuffle=False, num_workers=4)
    test_loader = NeighborSampler(pyg_data.edge_index, node_idx=test_mask, sizes=[10, 10], batch_size=32, shuffle=False, num_workers=4)
    return train_loader, valid_loader, test_loader

def train(model, train_loader, optimizer, criterion, device, pyg_data, full_train_labels):
    model.train()
    total_loss = 0
    for batch_size, n_id, adjs in train_loader:
        adjs = [adj.to(device) for adj in adjs]
        optimizer.zero_grad()
        out = model(pyg_data.x[n_id].to(device), adjs)
        loss = criterion(out, full_train_labels[n_id[:batch_size]].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, device, pyg_data, test_labels):
    model.eval()
    correct = 0
    total = 0
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        out = model(pyg_data.x[n_id].to(device), adjs)
        pred = out.argmax(dim=1)
        correct += int((pred == test_labels[n_id[:batch_size]].to(device)).sum())
        total += batch_size
    return correct / total

def main():
    pyg_data, train_labels, valid_labels, test_labels, train_set, valid_set, test_set, num_nodes = load_data()
    train_loader, valid_loader, test_loader = create_data_loaders(pyg_data, train_labels, valid_labels, test_labels, train_set, valid_set, test_set)
    
    full_train_labels = torch.full((num_nodes,), -1, dtype=train_labels.dtype)

    full_train_labels[train_set._items[0]] = train_labels

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(in_channels=pyg_data.x.size(1), hidden_channels=64, out_channels=train_labels.max().item() + 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(10):
        loss = train(model, train_loader, optimizer, criterion, device, pyg_data,full_train_labels)
        print(f'Epoch {epoch+1}, Loss: {loss}')

    accuracy = test(model, test_loader, device, pyg_data)
    print(f'Accuracy: {accuracy}')

if __name__ == "__main__":
    main()
