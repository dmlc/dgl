import argparse

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import DataLoader

import dgl
from dgl.nn.pytorch import GraphConv, SAGEConv
from dgl.dataloading.negative_sampler import GlobalUniform

from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


class NGNN_GCNConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(NGNN_GCNConv, self).__init__()
        self.conv = GraphConv(in_channels,  hidden_channels)
        self.fc_adj = Linear(hidden_channels, hidden_channels)
        self.fc_adj2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.fc_adj.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc_adj2.weight, gain=gain)
        for bias in [self.fc_adj.bias, self.fc_adj2.bias]:
            if bias is not None:
                import math
                stdv = 1.0 / math.sqrt(bias.size(0))
                bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)
        x = F.relu(x)
        x = self.fc_adj(x)
        x = F.relu(x)
        x = self.fc_adj2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, ngnn_type):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        if ngnn_type in ['input', 'all']:
            self.convs.append(NGNN_GCNConv(in_channels, hidden_channels, hidden_channels))
        else:
            self.convs.append(GraphConv(in_channels, hidden_channels))
            
        if ngnn_type in ['hidden', 'all']:
            for _ in range(num_layers - 2):
                self.convs.append(NGNN_GCNConv(hidden_channels, hidden_channels, hidden_channels))
        else:
            for _ in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
        
        if ngnn_type in ['output', 'all']:
            self.convs.append(NGNN_GCNConv(hidden_channels, hidden_channels, out_channels))
        else:
            self.convs.append(GraphConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class NGNN_SAGEConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 *, reduce):
        super(NGNN_SAGEConv, self).__init__()
        self.conv = SAGEConv(in_channels, hidden_channels, reduce)
        self.fc_adj = Linear(hidden_channels, hidden_channels)
        self.fc_adj2 = Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_uniform_(self.fc_adj.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc_adj2.weight, gain=gain)
        for bias in [self.fc_adj.bias, self.fc_adj2.bias]:
            if bias is not None:
                import math
                stdv = 1.0 / math.sqrt(bias.size(0))
                bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)
        x = F.relu(x)
        x = self.fc_adj(x)
        x = F.relu(x)
        x = self.fc_adj2(x)
        return x

class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, ngnn_type, reduce = 'mean'):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()

        if ngnn_type in ['input', 'all']:
            self.convs.append(NGNN_SAGEConv(in_channels, hidden_channels, hidden_channels, reduce=reduce))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, reduce))
        
        if ngnn_type in ['hidden', 'all']:
            for _ in range(num_layers - 2):
                self.convs.append(NGNN_SAGEConv(hidden_channels, hidden_channels, hidden_channels, reduce=reduce))
        else:
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, reduce))
        
        if ngnn_type in ['output', 'all']:
            self.convs.append(NGNN_SAGEConv(hidden_channels, hidden_channels, out_channels, reduce=reduce))
        else:
            self.convs.append(SAGEConv(hidden_channels, out_channels, reduce))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, g, x, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge['train']['edge'].to(x.device)
    neg_sampler = GlobalUniform(1)
    total_loss = total_examples = 0
    for perm in DataLoader(range(pos_train_edge.size(0)), batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = neg_sampler(g, edge[0])

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, g, valg, x, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(g, x)

    pos_train_edge = split_edge['eval_train']['edge'].to(h.device)
    pos_valid_edge = split_edge['valid']['edge'].to(h.device)
    neg_valid_edge = split_edge['valid']['edge_neg'].to(h.device)
    pos_test_edge = split_edge['test']['edge'].to(h.device)
    neg_test_edge = split_edge['test']['edge_neg'].to(h.device)

    def get_pred(test_edges, h):
        preds = []
        for perm in DataLoader(range(test_edges.size(0)), batch_size):
            edge = test_edges[perm].t()
            preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred
    
    pos_train_pred = get_pred(pos_train_edge, h)
    pos_valid_pred = get_pred(pos_valid_edge, h)
    neg_valid_pred = get_pred(neg_valid_edge, h)

    h = model(valg, x)

    pos_test_pred = get_pred(pos_test_edge, h)
    neg_test_pred = get_pred(neg_test_edge, h)

    results = {}
    for K in [10, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval({
            'y_pred_pos': pos_train_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_valid_pred,
            'y_pred_neg': neg_valid_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(description='OGBL-COLLAB (GNN)')

    # device settings
    parser.add_argument('--device', type=int, default=0, help='GPU device ID. Use -1 for CPU training.')

    # model structure settings
    parser.add_argument('--use_sage', action='store_true', help='If not set, use GCN by default.')
    parser.add_argument('--ngnn_type', type=str, default="none", choices=['none', 'input', 'hidden', 'output', 'all'], help="You can set this value from 'none', 'input', 'hidden' or 'all' to apply NGNN to different GNN layers.")
    parser.add_argument('--num_layers', type=int, default=3, help='number of GNN layers')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--batch_size', type=int, default=64 * 1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)

    # training settings
    parser.add_argument('--use_valedges_as_input', action='store_true', help='Use training + validation edges for inference on test set.')
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if args.device != -1 and torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglLinkPropPredDataset(name='ogbl-collab')
    data = dataset[0]
    split_edge = dataset.get_edge_split()

    n_feat = data.ndata['feat'].size(-1)
    
    # We randomly pick some training samples that we want to evaluate on:
    idx = torch.randperm(split_edge['train']['edge'].size(0))
    idx = idx[:split_edge['valid']['edge'].size(0)]
    split_edge['eval_train'] = {'edge': split_edge['train']['edge'][idx]}

    if args.use_sage:
        model = SAGE(n_feat, args.hidden_channels,
                     args.hidden_channels, args.num_layers,
                     args.dropout, args.ngnn_type).to(device)
    else: # GCN
        data = dgl.add_self_loop(data)
        model = GCN(n_feat, args.hidden_channels,
                    args.hidden_channels, args.num_layers,
                    args.dropout, args.ngnn_type).to(device)


    # Use training + validation edges for inference on test set.
    if args.use_valedges_as_input:
        val_edges = split_edge['valid']['edge'].T
        valdata = dgl.add_edges(data, val_edges[0], val_edges[1], {key: split_edge['valid'][key].view(-1, 1) for key in ['weight', 'year']})
        valdata = dgl.add_edges(valdata, val_edges[1], val_edges[0], {key: split_edge['valid'][key].view(-1, 1) for key in ['weight', 'year']})
    else:
        valdata = data

    data = data.to(device)
    valdata = valdata.to(device)

    predictor = LinkPredictor(args.hidden_channels, args.hidden_channels, 1, 3, args.dropout).to(device)

    evaluator = Evaluator(name='ogbl-collab')
    loggers = {
        'Hits@10': Logger(args.runs, args),
        'Hits@50': Logger(args.runs, args),
        'Hits@100': Logger(args.runs, args),
    }

    for run in range(args.runs):
        torch.cuda.empty_cache()
        model.reset_parameters()
        predictor.reset_parameters()
        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(predictor.parameters()),
            lr=args.lr)

        for epoch in range(1, 1 + args.epochs):
            loss = train(model, predictor, data, data.ndata['feat'], split_edge, optimizer,
                         args.batch_size)

            if epoch % args.eval_steps == 0:
                results = test(model, predictor, data, valdata, data.ndata['feat'], split_edge, evaluator,
                               args.batch_size)
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                if epoch % args.log_steps == 0:
                    for key, result in results.items():
                        train_hits, valid_hits, test_hits = result
                        print(key)
                        print(f'Run: {run + 1:02d}, '
                              f'Epoch: {epoch:02d}, '
                              f'Loss: {loss:.4f}, '
                              f'Train: {100 * train_hits:.2f}%, '
                              f'Valid: {100 * valid_hits:.2f}%, '
                              f'Test: {100 * test_hits:.2f}%')
                    print('---')

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
