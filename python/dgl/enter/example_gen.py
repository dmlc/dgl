import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl

class GCN(nn.Module):
    def __init__(self,
                 in_size,
                 out_size,
                 hidden_size: int = 16,
                 num_layers: int = 1,
                 norm: str = "both",
                 activation: str = "relu",
                 dropout: float = 0.5,
                 use_edge_weight: bool = False):
        """Graph Convolutional Networks

        Parameters
        ----------
        in_size : int
            Number of input features.
        out_size : int
            Output size.
        hidden_size : int
            Hidden size.
        num_layers : int
            Number of layers.
        norm : str
            GCN normalization type. Can be 'both', 'right', 'left', 'none'.
        activation : str
            Activation function.
        dropout : float
            Dropout rate.
        use_edge_weight : bool
            If true, scale the messages by edge weights.
        """
        super().__init__()
        self.use_edge_weight = use_edge_weight
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(dgl.nn.GraphConv(in_size, hidden_size, norm=norm))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(dgl.nn.GraphConv(hidden_size, hidden_size, norm=norm))
        # output layer
        self.layers.append(dgl.nn.GraphConv(hidden_size, out_size, norm=norm))
        self.dropout = nn.Dropout(p=dropout)
        self.act = getattr(torch, activation)

    def forward(self, g, node_feat, edge_feat):
        h = node_feat
        edge_weight = edge_feat if self.use_edge_weight else None
        for layer in self.layers[:-1]:
            h = layer(g, h, edge_weight=edge_weight)
            h = self.act(h)
            h = self.dropout(h)
        h = self.layers[-1](g, h, edge_weight=edge_weight)
        return h

class EarlyStopping:
    def __init__(self,
                 patience: int = -1,
                 checkpoint_path: str = 'checkpoint.pt'):
        self.patience = patience
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def step(self, acc, model):
        score = acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        '''Save model when validation loss decreases.'''
        torch.save(model.state_dict(), self.checkpoint_path)

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def train(cfg, device, data, model, optimizer, loss_fcn):
    g = data[0]  # Only train on the first graph
    g = g.to(device)
    node_feat = g.ndata['feat']
    edge_feat = g.edata.get('feat', None)
    label = g.ndata['label']
    train_mask, val_mask, test_mask = g.ndata['train_mask'], g.ndata['val_mask'], g.ndata['test_mask']

    stopper = EarlyStopping(cfg['patience'], cfg['checkpoint_path'])
    val_acc = 0.
    for epoch in range(cfg['num_epochs']):
        model.train()
        logits = model(g, node_feat, edge_feat)
        loss = loss_fcn(logits[train_mask], label[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc = accuracy(logits[train_mask], label[train_mask])
        if epoch != 0 and epoch % cfg['eval_period'] == 0:
            val_acc = accuracy(logits[val_mask], label[val_mask])
            if stopper.step(val_acc, model):
                break
        print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f} | ValAcc {:.4f}".
              format(epoch, loss.item(), train_acc, val_acc))

    stopper.load_checkpoint(model)
    model.eval()
    with torch.no_grad():
        logits = model(g, node_feat, edge_feat)
        test_acc = accuracy(logits[test_mask], label[test_mask])
    print("Test Accuracy {:.4f}".format(test_acc))

class GNNWithEmbedding(nn.Module):
    def __init__(self, num_nodes, embed_size, gnn):
        super().__init__()
        self.embed = nn.Embedding(num_nodes, embed_size)
        self.gnn = gnn

    def forward(self, g, node_feat, edge_feat):
        return self.gnn(g, self.embed.weight, edge_feat)

def main():
    cfg = {
        'node_embed_size' : -1,
        'num_epochs' : 200,
        'eval_period' : 5,
        'checkpoint_path' : 'checkpoint.pt',
        'patience' : 20,
        'device': 'cuda:0'
    }
    device = cfg['device']
    data = dgl.data.CoraGraphDataset()
    feat_size = data[0].ndata['feat'].shape[1]
    in_size = cfg['node_embed_size'] if cfg['node_embed_size'] > 0 else feat_size
    out_size = data.num_classes
    model = GCN(in_size,
                out_size,
                hidden_size=16,
                num_layers=1,
                norm="both",
                activation="relu",
                dropout=0.5,
                use_edge_weight=False)
    if cfg['node_embed_size'] > 0:
        model = GNNWithEmbedding(data[0].num_nodes(), cfg['node_embed_size'], model)
    model = model.to(device)
    params = model.parameters()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, lr=1e-2, weight_decay=5e-4)
    train(cfg, device, data, model, optimizer, loss)

if __name__ == '__main__':
    main()
