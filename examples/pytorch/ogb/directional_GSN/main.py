from ogb.graphproppred import Evaluator
import torch
import numpy as np
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
import dgl
import random
import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder
import torch.nn.functional as F
from functools import partial
import torch.optim as optim
import argparse
from torch.utils.data import Dataset
from dgl.data.utils import Subset
from preprocessing import prepare_dataset

def aggregate_mean(h, vector_field, h_in):
    return torch.mean(h, dim=1)

def aggregate_max(h, vector_field, h_in):
    return torch.max(h, dim=1)[0]

def aggregate_sum(h, vector_field, h_in):
    return torch.sum(h, dim=1)

def aggregate_dir_dx(h, vector_field, h_in, eig_idx):
    eig_w = ((vector_field[:, :, eig_idx]) /
             (torch.sum(torch.abs(vector_field[:, :, eig_idx]), keepdim=True, dim=1) + 1e-8)).unsqueeze(-1)
    h_mod = torch.mul(h, eig_w)
    return torch.abs(torch.sum(h_mod, dim=1) - torch.sum(eig_w, dim=1) * h_in)

AGGREGATORS = {'mean': aggregate_mean, 'sum': aggregate_sum, 'max': aggregate_max,
               'dir1-dx': partial(aggregate_dir_dx, eig_idx=1)}

class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, 
                 dropout=0., last_b_norm=False, device='cpu'):
        super(MLP, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size

        self.fully_connected = nn.ModuleList()
        self.fully_connected.append(FCLayer(in_size, out_size, b_norm=last_b_norm,
                                            device=device, dropout=dropout))

    def forward(self, x):
        for fc in self.fully_connected:
            x = fc(x)
        return x

class FCLayer(nn.Module):
    def __init__(self, in_size, out_size, device, dropout=0., b_norm=False, bias=True):
        super(FCLayer, self).__init__()

        self.__params = locals()
        del self.__params['__class__']
        del self.__params['self']
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.linear = nn.Linear(in_size, out_size, bias=bias).to(device)
        self.dropout = None
        self.b_norm = None
        if dropout:
            self.dropout = nn.Dropout(p=dropout, device=device)
        if b_norm:
            self.b_norm = nn.BatchNorm1d(out_size).to(device)
        self.init_fn = nn.init.xavier_uniform_

        self.reset_parameters()

    def reset_parameters(self, init_fn=None):
        init_fn = init_fn or self.init_fn
        if init_fn is not None:
            init_fn(self.linear.weight, 1 / self.in_size)
        if self.bias:
            self.linear.bias.data.zero_()

    def forward(self, x):
        h = self.linear(x)
        if self.dropout is not None:
            h = self.dropout(h)
        if self.b_norm is not None:
            if h.shape[1] != self.out_size:
                h = self.b_norm(h.transpose(1, 2)).transpose(1, 2)
            else:
                h = self.b_norm(h)
        return h

class DGNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout, aggregators, avg_d, residual):
        super().__init__()

        self.dropout = dropout
        self.residual = residual

        self.aggregators = aggregators

        self.batchnorm_h = nn.BatchNorm1d(out_dim)
        self.pretrans = MLP(in_size=2 * in_dim, hidden_size=in_dim, out_size=in_dim)
        self.posttrans = MLP(in_size=(len(aggregators) * 1 + 1) * in_dim, hidden_size=out_dim,
                             out_size=out_dim)
        self.avg_d = avg_d
        if in_dim != out_dim:
            self.residual = False

    def pretrans_edges(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        vector_field = edges.data['eig']
        return {'e': self.pretrans(z2), 'vector_field': vector_field}

    def message_func(self, edges):
        return {'e': edges.data['e'], 'vector_field': edges.data['vector_field'].to('cuda' if torch.cuda.is_available() else 'cpu')}

    def reduce_func(self, nodes):
        h_in = nodes.data['h']
        h = nodes.mailbox['e']

        vector_field = nodes.mailbox['vector_field']

        h = torch.cat([aggregate(h, vector_field, h_in) for aggregate in self.aggregators], dim=1)

        return {'h': h}

    def posttrans_nodes(self, nodes):
        return self.posttrans(nodes.data['h'])

    def forward(self, g, h, e, snorm_n):

        h_in = h
        g.ndata['h'] = h

        # pretransformation
        g.apply_edges(self.pretrans_edges)

        # aggregation
        g.update_all(self.message_func, self.reduce_func)
        h = torch.cat([h, g.ndata['h']], dim=1)

        # posttransformation
        h = self.posttrans(h)

        # graph and batch normalization and residual\
        h = h * snorm_n
        h = self.batchnorm_h(h)
        h = F.relu(h)
        if self.residual:
            h = h_in + h

        h = F.dropout(h, self.dropout, training=self.training)

        return h

class MLPReadout(nn.Module):

    def __init__(self, input_dim, output_dim, L=2):  # L=nb_hidden_layers
        super().__init__()
        list_FC_layers = [nn.Linear(input_dim // 2 ** l, input_dim // 2 ** (l + 1), bias=True) for l in range(L)]
        list_FC_layers.append(nn.Linear(input_dim // 2 ** L, output_dim, bias=True))
        self.FC_layers = nn.ModuleList(list_FC_layers)
        self.L = L

    def forward(self, x):
        y = x
        for l in range(self.L):
            y = self.FC_layers[l](y)
            y = F.relu(y)
        y = self.FC_layers[self.L](y)
        return y

class DGNNet(nn.Module):
    def __init__(self, hidden_dim, out_dim, dropout, n_layers, avg_d):
        super().__init__()
        self.aggregators = "mean sum max dir1-dx"
        self.avg_d = avg_d
        self.residual = False

        self.embedding_h = AtomEncoder(emb_dim=hidden_dim)

        # retrieve the aggregators and scalers functions
        self.aggregators = [AGGREGATORS[aggr] for aggr in self.aggregators.split()]

        self.layers = nn.ModuleList([DGNLayer(in_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout, 
                      residual=self.residual, aggregators=self.aggregators,
                      avg_d=self.avg_d) for _ in range(n_layers - 1)])
        self.layers.append(DGNLayer(in_dim=hidden_dim, out_dim=out_dim, dropout=dropout,
                                    residual=self.residual, aggregators=self.aggregators, 
                                    avg_d=self.avg_d))

        # 128 out dim since ogbg-molpcba has 128 tasks
        self.MLP_layer = MLPReadout(out_dim, 128)

    def forward(self, g, h, e, snorm_n):
        h = self.embedding_h(h)

        for i, conv in enumerate(self.layers):
            h_t = conv(g, h, e, snorm_n)
            h = h_t

        g.ndata['h'] = h

        hg = dgl.mean_nodes(g, 'h')

        return self.MLP_layer(hg)

    def loss(self, scores, labels):
        is_labeled = labels == labels
        loss = torch.nn.BCEWithLogitsLoss()(scores[is_labeled], labels[is_labeled].float().to('cuda'))
        return loss

def train_epoch(model, optimizer, device, data_loader):
    model.train()
    epoch_loss = 0
    epoch_train_AP = 0
    list_scores = []
    list_labels = []
    for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_snorm_e = batch_snorm_e.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        batch_labels = batch_labels.to(device)
        batch_graphs = batch_graphs.to(device)
        optimizer.zero_grad()

        batch_scores = model(batch_graphs, batch_x, batch_e, batch_snorm_n)
        
        loss = model.loss(batch_scores, batch_labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        list_scores.append(batch_scores.detach())
        list_labels.append(batch_labels.detach())

    epoch_loss /= (iter + 1)

    evaluator = Evaluator(name='ogbg-molpcba')
    epoch_train_AP = evaluator.eval({'y_pred': torch.cat(list_scores),
                                        'y_true': torch.cat(list_labels)})['ap']

    return epoch_loss, epoch_train_AP, optimizer

def evaluate_network(model, device, data_loader):
    model.eval()
    epoch_test_loss = 0
    epoch_test_AP = 0
    with torch.no_grad():
        list_scores = []
        list_labels = []
        for iter, (batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e) in enumerate(data_loader):
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_snorm_n = batch_snorm_n.to(device)
            batch_labels = batch_labels.to(device)
            batch_graphs = batch_graphs.to(device)

            batch_scores = model.forward(batch_graphs, batch_x, batch_e, batch_snorm_n)

            loss = model.loss(batch_scores, batch_labels)
            epoch_test_loss += loss.detach().item()
            list_scores.append(batch_scores.detach())
            list_labels.append(batch_labels.detach())

        epoch_test_loss /= (iter + 1)

        evaluator = Evaluator(name='ogbg-molpcba')
        epoch_test_AP = evaluator.eval({'y_pred': torch.cat(list_scores),
                                           'y_true': torch.cat(list_labels)})['ap']

    return epoch_test_loss, epoch_test_AP

def view_model_param(net_params):
    model = DGNNet(net_params.hidden_dim, net_params.out_dim, net_params.dropout, net_params.L, net_params.avg_d)
    total_param = 0
    print("MODEL DETAILS:\n")
    for param in model.parameters():
        total_param += np.prod(list(param.data.size()))
    print('DGN Total parameters:', total_param)
    return total_param

def train(dataset, params):

    trainset, valset, testset = dataset.train, dataset.val, dataset.test
    device = params.device

    print("Training Graphs: ", len(trainset))
    print("Validation Graphs: ", len(valset))
    print("Test Graphs: ", len(testset))

    model = DGNNet(params.hidden_dim, params.out_dim, params.dropout, params.L, params.avg_d)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.init_lr, weight_decay=params.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=params.lr_reduce_factor,
                                                     patience=params.lr_schedule_patience,
                                                     verbose=True)

    epoch_train_losses, epoch_val_losses = [], []
    epoch_train_APs, epoch_val_APs, epoch_test_APs = [], [], []

    train_loader = GraphDataLoader(trainset, batch_size=params.batch_size, shuffle=True, collate_fn=dataset.collate, pin_memory=True)
    val_loader = GraphDataLoader(valset, batch_size=params.batch_size, shuffle=False, collate_fn=dataset.collate, pin_memory=True)
    test_loader = GraphDataLoader(testset, batch_size=params.batch_size, shuffle=False, collate_fn=dataset.collate, pin_memory=True)

    with tqdm(range(params.epochs), unit='epoch') as t:
        for epoch in t:
            t.set_description('Epoch %d' % epoch)

            epoch_train_loss, epoch_train_ap, optimizer = train_epoch(model, optimizer, device, train_loader)
            epoch_val_loss, epoch_val_ap = evaluate_network(model, device, val_loader)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            epoch_train_APs.append(epoch_train_ap.item())
            epoch_val_APs.append(epoch_val_ap.item())

            _, epoch_test_ap = evaluate_network(model, device, test_loader)

            epoch_test_APs.append(epoch_test_ap.item())

            t.set_postfix(train_loss=epoch_train_loss, 
                          train_AP=epoch_train_ap.item(), val_AP=epoch_val_ap.item(),
                          refresh=False)

            scheduler.step(-epoch_val_ap.item())

            if optimizer.param_groups[0]['lr'] < params.min_lr:
                print("\n!! LR EQUAL TO MIN LR SET.")
                break

            print('')

    best_val_epoch = np.argmax(np.array(epoch_val_APs))
    best_train_epoch = np.argmax(np.array(epoch_train_APs))
    best_val_ap = epoch_val_APs[best_val_epoch]
    best_val_test_ap = epoch_test_APs[best_val_epoch]
    best_val_train_ap = epoch_train_APs[best_val_epoch]
    best_train_ap = epoch_train_APs[best_train_epoch]

    print("Best Train AP: {:.4f}".format(best_train_ap))
    print("Best Val AP: {:.4f}".format(best_val_ap))
    print("Test AP of Best Val: {:.4f}".format(best_val_test_ap))
    print("Train AP of Best Val: {:.4f}".format(best_val_train_ap))

class PCBADataset(Dataset):
    def __init__(self, name):
        print("[I] Loading dataset %s..." % (name))
        self.name = name
        
        self.dataset, self.split_idx = prepare_dataset(name)
        print("One hot encoding substructure counts... ", end='')
        self.d_id = [1]*self.dataset[0][0].edata['subgraph_counts'].shape[1]

        for g in self.dataset:
            g[0].edata['eig'] = g[0].edata['subgraph_counts'].float()
            
        self.raw_train = Subset(self.dataset, self.split_idx['train'])
        self.raw_val = Subset(self.dataset, self.split_idx['valid'])
        self.raw_test = Subset(self.dataset, self.split_idx['test'])

        self.train, self.val, self.test = [], [], []
        for g in self.raw_train:
            if g[0].num_nodes() > 5:
                self.train.append(g)
        for g in self.raw_val:
            if g[0].num_nodes() > 5:
                self.val.append(g)
        for g in self.raw_test:
            if g[0].num_nodes() > 5:
                self.test.append(g)

        print('train, test, val sizes :', len(self.train), len(self.test), len(self.val))
        print("[I] Finished loading.")

    # form a mini batch from a given list of samples = [(graph, label) pairs]
    def collate(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.stack(labels)

        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()
        tab_sizes_e = [ graphs[i].number_of_edges() for i in range(len(graphs))]
        tab_snorm_e = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_e ]
        snorm_e = torch.cat(tab_snorm_e).sqrt()
        batched_graph = dgl.batch(graphs)

        return batched_graph, labels, snorm_n, snorm_e

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int, help="Please give a value for gpu id")
    parser.add_argument('--seed', default=41, type=int, help="Please give a value for seed")
    parser.add_argument('--epochs', default=450, type=int, help="Please give a value for epochs")
    parser.add_argument('--batch_size', default=2048, type=int, help="Please give a value for batch_size")
    parser.add_argument('--init_lr', default=0.0008, type=float, help="Please give a value for init_lr")
    parser.add_argument('--lr_reduce_factor', default=0.8, type=float, help="Please give a value for lr_reduce_factor")
    parser.add_argument('--lr_schedule_patience', default=8, type=int, help="Please give a value for lr_schedule_patience")
    parser.add_argument('--min_lr', default=0.00001, type=float, help="Please give a value for min_lr")
    parser.add_argument('--weight_decay', default=1e-5, type=float, help="Please give a value for weight_decay")
    parser.add_argument('--L', default=4, type=int, help="Please give a value for L")
    parser.add_argument('--hidden_dim', default=420, type=int, help="Please give a value for hidden_dim")
    parser.add_argument('--out_dim', default=420, type=int, help="Please give a value for out_dim")
    parser.add_argument('--dropout', default=0.2, type=float, help="Please give a value for dropout")
    args = parser.parse_args()

    # device
    args.device = torch.device("cuda:{}".format(args.gpu_id))
        
    # setting seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset = PCBADataset("ogbg-molpcba")
    train_graph_lists = []
    for g in dataset.train:
        train_graph_lists.append(g[0])
    D = torch.cat([torch.sparse.sum(g.adjacency_matrix(transpose=True), dim=-1).to_dense() for g in
                       train_graph_lists])
    args.avg_d = dict(lin=torch.mean(D),
                 exp=torch.mean(torch.exp(torch.div(1, D)) - 1),
                 log=torch.mean(torch.log(D + 1)))

    view_model_param(args)
    train(dataset, args)