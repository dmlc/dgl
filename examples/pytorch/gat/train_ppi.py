import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader
from sklearn.metrics import f1_score

class GAT(nn.Module):
    def __init__(self,in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        # three-layer GAT
        self.gat_layers.append(dglnn.GATConv(in_size, hid_size, heads[0], activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[0], hid_size, heads[1], residual=True, activation=F.elu))
        self.gat_layers.append(dglnn.GATConv(hid_size*heads[1], out_size, heads[2], residual=True, activation=None))
        
    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.gat_layers):
            h = layer(g, h)
            if i == 2:  # last layer 
                h = h.mean(1)
            else:       # other layer(s)
                h = h.flatten(1)
        return h

def evaluate(g, features, labels, model):
    model.eval()
    with torch.no_grad():
        for layer in model.gat_layers:
            layer.g = g
        output = model(g, features)
        pred = np.where(output.data.cpu().numpy() >= 0, 1, 0)
        score = f1_score(labels.data.cpu().numpy(), pred, average='micro')
        return score

def evaluate_in_batches(dataloader, device, model):
    total_score = 0
    for batch_id, subgraph in enumerate(dataloader):
        subgraph = subgraph.to(device)
        features = subgraph.ndata['feat']
        labels = subgraph.ndata['label']
        score = evaluate(subgraph, features, labels, model)
        total_score += score
    return total_score / (batch_id + 1) # return average score
    
def train(train_dataloader, valid_dataloader, features, device, model):
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3, weight_decay=0)

    # training loop        
    for epoch in range(400):
        model.train()
        logits = []
        total_loss = 0.0
        # mini-batches loop
        for batch_id, subgraph in enumerate(train_dataloader):
            subgraph = subgraph.to(device)
            features = subgraph.ndata['feat'].float()
            labels = subgraph.ndata['label'].float()
            for layer in model.gat_layers:
                layer.g = subgraph
            logits = model(subgraph, features)
            loss = loss_fcn(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print("Epoch {:05d} | Loss {:.4f} |". format(epoch, total_loss / (batch_id + 1) ))
        
        if (epoch + 1) % 5 == 0:
            avg_score = evaluate_in_batches(valid_dataloader, device, model) # evaluate F1-score instead of loss
            print("                            Acc. (F1-score) {:.4f} ". format(avg_score))

        
if __name__ == '__main__':
    print(f'Training PPI Dataset with DGL built-in GATConv module.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # load and preprocess datasets
    train_dataset = PPIDataset(mode='train')
    valid_dataset = PPIDataset(mode='valid')
    test_dataset = PPIDataset(mode='test')
    features = train_dataset[0].ndata['feat']
    
    # create GAT model    
    in_size = features.shape[1]
    out_size = train_dataset.num_labels
    model = GAT(in_size, 256, out_size, heads=[4,4,6]).to(device)
    
    # model training
    print('Training...')
    train_dataloader = GraphDataLoader(train_dataset, batch_size=2)
    valid_dataloader = GraphDataLoader(valid_dataset, batch_size=2)
    train(train_dataloader, valid_dataloader, features, device, model)

    # test the model
    print('Testing...')
    test_dataloader = GraphDataLoader(test_dataset, batch_size=2)
    avg_score = evaluate_in_batches(test_dataloader, device, model) # evaluate in F1-Score instead of loss
    print("Test Accuracy (F1-score) {:.4f}".format(avg_score))
