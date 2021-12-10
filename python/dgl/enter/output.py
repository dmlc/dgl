import dgl
import dgl.nn as dglnn
import torch
import torch as th
import torch.functional as F
import torch.nn as nn
import torch.nn.functional as F
import tqdm


class GCN(nn.Module):
    def __init__(self, in_size,
                 out_size,
                 hidden_size: int = 16,
                 num_layers: int = 2,
                 activation=F.relu,
                 dropout: float = 0.5
                 ):
        super().__init__()
        self.init(
            in_size,
            hidden_size,
            out_size,
            num_layers,
            activation,
            dropout)

    def init(
            self,
            in_size,
            hidden_size,
            out_size,
            num_layers,
            activation,
            dropout):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.layers = nn.ModuleList()
        if num_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_size, hidden_size, 'mean'))
            for i in range(1, num_layers - 1):
                self.layers.append(
                    dglnn.SAGEConv(
                        hidden_size,
                        hidden_size,
                        'mean'))
            self.layers.append(dglnn.SAGEConv(hidden_size, out_size, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        for l, layer in enumerate(self.layers):
            y = th.zeros(
                g.num_nodes(),
                self.hidden_size if l != len(
                    self.layers) -
                1 else self.out_size)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                device=device if num_workers == 0 else None,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


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
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
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


def load_subtensor(nfeat, labels, seeds, input_nodes, device):
    """
    Extracts features and labels for a subset of nodes
    """
    batch_inputs = nfeat[input_nodes].to(device)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels


def evaluate(model, g, nfeat, labels, val_nid, device, cfg):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    device : The GPU device to evaluate on.
    """
    model.eval()
    batch_size = 1
    num_workers = 0
    with torch.no_grad():
        pred = model.inference(g, nfeat, device, batch_size, num_workers)
    model.train()
    return accuracy(pred[val_nid], labels[val_nid].to(pred.device))


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)


def train(cfg, device, data, model, optimizer, loss_fcn):
    g = data[0]  # Only train on the first graph
    g = g.to(device)
    train_g = val_g = test_g = g
    train_nfeat = val_nfeat = test_nfeat = train_g.ndata['feat']
    train_labels = val_labels = test_labels = train_g.ndata['label']

    train_nid = torch.nonzero(train_g.ndata['train_mask'], as_tuple=True)[0]
    val_nid = torch.nonzero(val_g.ndata['val_mask'], as_tuple=True)[0]
    test_nid = torch.nonzero(
        ~(test_g.ndata['train_mask'] | test_g.ndata['val_mask']), as_tuple=True)[0]

    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        [int(fanout) for fanout in cfg["fan_out"].split(',')])
    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=cfg["batch_size"],
        shuffle=True,
        drop_last=False,
        num_workers=cfg["num_workers"])

    stopper = EarlyStopping(cfg['patience'], cfg['checkpoint_path'])

    val_acc = 0.
    for epoch in range(cfg['num_epochs']):
        model.train()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(
                train_nfeat, train_labels, seeds, input_nodes, device)
            blocks = [block.int().to(device) for block in blocks]
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step != 0 and step % cfg['eval_period'] == 0:
                train_acc = accuracy(batch_pred, batch_labels)
                print("Epoch {:05d} | Loss {:.4f} | TrainAcc {:.4f}".
                      format(epoch, loss.item(), train_acc))

        if epoch % cfg["eval_every"] == 0 and epoch != 0:
            eval_acc = evaluate(
                model,
                val_g,
                val_nfeat,
                val_labels,
                val_nid,
                device,
                cfg)
            print('Eval Acc {:.4f}'.format(eval_acc))

        if stopper.step(val_acc, model):
            break

    stopper.load_checkpoint(model)

    model.eval()
    with torch.no_grad():
        test_acc = evaluate(
            model,
            test_g,
            test_nfeat,
            test_labels,
            test_nid,
            device, cfg)
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
        'device': 'cpu',
        'node_embed_size': -1,
        'model': {},
        'fan_out': '5,10',
        'batch_size': 32,
        'num_workers': 4,
        'early_stop': True,
        'num_epochs': 200,
        'eval_period': 5,
        'checkpoint_path': 'checkpoint.pt',
        'patience': 20,
        'optimizer': {
            'lr': 0.005,
            'weight_decay': 0},
        'loss': 'CrossEntropyLoss'}
    device = cfg['device']
    data = dgl.data.RedditDataset()
    g = data[0]
    feat_size = g.ndata['feat'].shape[1]
    cfg["model"]["in_size"] = cfg['node_embed_size'] if cfg['node_embed_size'] > 0 else feat_size
    cfg["model"]["out_size"] = data.num_classes
    model = GCN(**cfg["model"])
    if cfg['node_embed_size'] > 0:
        model = GNNWithEmbedding(
            data[0].num_nodes(),
            cfg['node_embed_size'],
            model)
    model = model.to(device)
    params = model.parameters()
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params, **cfg["optimizer"])
    train(cfg, device, data, model, optimizer, loss)


if __name__ == '__main__':
    main()
