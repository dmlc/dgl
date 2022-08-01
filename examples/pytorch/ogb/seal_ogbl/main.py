import argparse
import time
import os, sys
from tqdm import tqdm
import torch
import dgl
from torch.nn import BCEWithLogitsLoss
from torch.utils.data import Dataset
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import save_graphs, load_graphs
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from utils import *
from models import *


class SEALOGBLDataset(Dataset):
    def __init__(self, root, graph, split_edge, percent=100, split='train',
                 ratio_per_hop=1.0, directed=False, dynamic=True) -> None:
        super().__init__()
        self.root = root
        self.graph = graph
        self.split = split
        self.split_edge = split_edge
        self.percent = percent
        self.ratio_per_hop = ratio_per_hop
        self.directed = directed
        self.dynamic = dynamic

        if 'weights' in self.graph.edata:
            self.edge_weights = self.graph.edata['weights']
        else:
            self.edge_weights = None
        if 'feat' in self.graph.ndata:
            self.node_features = self.graph.ndata['feat']
        else:
            self.node_features = None

        if not self.dynamic:
            self.g_list, tensor_dict = self.load_cached()
            self.labels = tensor_dict['y']
            return

        pos_edge, neg_edge = get_pos_neg_edges(split, self.split_edge, self.graph, self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 0).tolist() # [Np + Nn, 2]
        self.labels = [1] * len(pos_edge) + [0] * len(neg_edge)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if not self.dynamic:
            g, y = self.g_list[idx], self.labels[idx]
            x = None if 'x' not in g.ndata else g.ndata['x']
            w = None if 'w' not in g.edata else g.eata['w']
            return g, g.ndata['z'], x, w, y

        src, dst = self.links[idx]
        y = self.labels[idx]
        subg = k_hop_subgraph(src, dst, 1, self.graph,
            self.ratio_per_hop, self.directed)

        # remove the link between src and dst
        direct_links = [[], []]
        for s, t in [(0, 1), (1, 0)]:
            if subg.has_edges_between(s, t):
                direct_links[0].append(s)
                direct_links[1].append(t)
        if len(direct_links[0]):
            subg.remove_edges(subg.edge_ids(*direct_links))

        NIDs, EIDs = subg.ndata[dgl.NID], subg.edata[dgl.EID]

        z = drnl_node_labeling(subg.adj(scipy_fmt='csr'), 0, 1)
        edge_weights = self.edge_weights[EIDs] if self.edge_weights is not None else None
        x = self.node_features[NIDs] if self.node_features is not None else None

        subg_aug = subg.add_self_loop()
        if edge_weights is not None:
            edge_weights = torch.cat([
                edge_weights, torch.ones(subg_aug.num_edges() - subg.num_edges())])
        return subg_aug, z, x, edge_weights, y

    @property
    def cached_name(self):
        return 'SEAL_{}_data_{}.pt'.format(self.split, self.percent)

    def process(self):
        g_list, labels = [], []
        self.dynamic = True
        for i in range(len(self)):
            g, z, x, weights, y = self[i]
            g.ndata['z'] = z
            if x is not None:
                g.ndata['x'] = x
            if weights is not None:
                g.edata['w'] = weights
            g_list.append(g)
            labels.append(y)
        self.dynamic = False
        return g_list, {'y': torch.tensor(labels)}

    def load_cached(self):
        path = os.path.join(self.root, self.cached_name)
        if os.path.exists(path):
            return load_graphs(path)

        if not os.path.exists(self.root):
            os.makedirs(self.root)

        pos_edge, neg_edge = get_pos_neg_edges(
            self.split, self.split_edge, self.graph, self.percent)
        self.links = torch.cat([pos_edge, neg_edge], 0).tolist() # [Np + Nn, 2]
        self.labels = [1] * len(pos_edge) + [0] * len(neg_edge)

        g_list, labels = self.process()
        save_graphs(path, g_list, labels)
        return g_list, labels


def ogbl_collate_fn(batch):
    gs, zs, xs, ws, ys = zip(*batch)
    batched_g = dgl.batch(gs)
    z = torch.cat(zs, dim=0)
    if xs[0] is not None:
        x = torch.cat(xs, dim=0)
    else:
        x = None
    if ws[0] is not None:
        edge_weights = torch.cat(ws, dim=0)
    else:
        edge_weights = None
    y = torch.tensor(ys)

    return batched_g, z, x, edge_weights, y


def train():
    model.train()
    loss_fnt = BCEWithLogitsLoss()
    total_loss = 0
    pbar = tqdm(train_loader, ncols=70)
    for batch in pbar:
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in batch]
        optimizer.zero_grad()
        logits = model(g, z, x, edge_weight=edge_weights)
        loss = loss_fnt(logits.view(-1), y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * g.batch_size

    return total_loss / len(train_dataset)


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for batch in tqdm(val_loader, ncols=70):
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in batch]
        logits = model(g, z, x, edge_weight=edge_weights)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true==1]
    neg_val_pred = val_pred[val_true==0]

    y_pred, y_true = [], []
    for batch in tqdm(test_loader, ncols=70):
        g, z, x, edge_weights, y = [
            item.to(device) if item is not None else None for item in batch]
        logits = model(g, z, x, edge_weight=edge_weights)
        y_pred.append(logits.view(-1).cpu())
        y_true.append(y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true==1]
    neg_test_pred = test_pred[test_true==0]
    
    if args.eval_metric == 'hits':
        results = evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)
    elif args.eval_metric == 'mrr':
        results = evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred)

    return results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval({
            'y_pred_pos': pos_val_pred,
            'y_pred_neg': neg_val_pred,
        })[f'hits@{K}']
        test_hits = evaluator.eval({
            'y_pred_pos': pos_test_pred,
            'y_pred_neg': neg_test_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = (valid_hits, test_hits)

    return results
     

def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    print(pos_val_pred.size(), neg_val_pred.size(), pos_test_pred.size(), neg_test_pred.size())
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = evaluator.eval({
        'y_pred_pos': pos_val_pred,
        'y_pred_neg': neg_val_pred,
    })['mrr_list'].mean().item()

    test_mrr = evaluator.eval({
        'y_pred_pos': pos_test_pred,
        'y_pred_neg': neg_test_pred,
    })['mrr_list'].mean().item()

    results['MRR'] = (valid_mrr, test_mrr)
    
    return results


if __name__ == '__main__':
    # Data settings
    parser = argparse.ArgumentParser(description='OGBL (SEAL)')
    parser.add_argument('--dataset', type=str, default='ogbl-collab')
    # GNN settings
    parser.add_argument('--sortpool_k', type=float, default=0.6)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    # Subgraph extraction settings
    parser.add_argument('--ratio_per_hop', type=float, default=1.0)
    parser.add_argument('--use_feature', action='store_true', 
                        help="whether to use raw node features as GNN input")
    parser.add_argument('--use_edge_weight', action='store_true', 
                        help="whether to consider edge weight in GNN")
    # Training settings
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--train_percent', type=float, default=100)
    parser.add_argument('--val_percent', type=float, default=100)
    parser.add_argument('--test_percent', type=float, default=100)
    parser.add_argument('--dynamic_train', action='store_true', 
                        help="dynamically extract enclosing subgraphs on the fly")
    parser.add_argument('--dynamic_val', action='store_true')
    parser.add_argument('--dynamic_test', action='store_true')
    parser.add_argument('--num_workers', type=int, default=0, 
                        help="number of workers for dynamic dataloaders; using a larger"
                        + " value for dynamic dataloading is recommended")
    # Testing settings
    parser.add_argument('--use_valedges_as_input', action='store_true')
    parser.add_argument('--eval_steps', type=int, default=1)
    args = parser.parse_args()

    data_appendix = '_rph{}'.format(''.join(str(args.ratio_per_hop).split('.')))
    if args.use_valedges_as_input:
        data_appendix += '_uvai'

    args.res_dir = os.path.join('results/{}_{}'.format(args.dataset,
        time.strftime("%Y%m%d%H%M%S")))
    print('Results will be saved in ' + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir) 
    log_file = os.path.join(args.res_dir, 'log.txt')
    # Save command line input.
    cmd_input = 'python ' + ' '.join(sys.argv) + '\n'
    with open(os.path.join(args.res_dir, 'cmd_input.txt'), 'a') as f:
        f.write(cmd_input)
    print('Command line input: ' + cmd_input + ' is saved.')
    with open(log_file, 'a') as f:
        f.write('\n' + cmd_input)

    dataset = DglLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # re-format the data of citation2
    if args.dataset == 'ogbl-citation2':
        for k in ['train', 'valid', 'test']:
            src = split_edge[k]['source_node']
            tgt = split_edge[k]['target_node']
            split_edge[k]['edge'] = torch.stack([src, tgt], dim=1)
            if k != 'train':
                tgt_neg = split_edge[k]['target_node_neg']
                split_edge[k]['edge_neg'] = torch.stack([
                    src[:, None].repeat(1, tgt_neg.size(1)),
                    tgt_neg
                ], dim=-1)  # [Ns, Nt, 2]

    # reconstruct the graph for ogbl-collab data for validation edge augmentation and coalesce
    if args.dataset == 'ogbl-collab':
        edges = torch.stack(graph.edges(), dim=1)  # [L, 2]
        weights = graph.edata['weight'].squeeze()

        # use valid edges as input for ogbl-collab
        if args.use_valedges_as_input:
            val_edges = split_edge['valid']['edge']
            # to undirected
            row, col = val_edges.t()
            val_edges = torch.stack([
                torch.cat([row, col], dim=0),
                torch.cat([col, row], dim=0)
            ], dim=1)  # [2*L, 2]
            val_weights = torch.ones(size=(val_edges.size(0),), dtype=int)
            edges = torch.cat([edges, val_edges], dim=0)
            weights = torch.cat([weights, val_weights])

        # coalesce
        coo_m = torch.sparse_coo_tensor(edges.t(), weights).coalesce()
        coo_m = coo_m.coalesce()  # [2, L]
        graph = dgl.graph(tuple(coo_m.indices()))
        graph.edata['weight'] = coo_m.values()

    if not args.use_edge_weight and 'weight' in graph.edata:
        del graph.edata['weight']
    if not args.use_feature and 'feat' in graph.ndata:
        del graph.ndata['feat']

    if args.dataset.startswith('ogbl-citation'):
        args.eval_metric = 'mrr'
        directed = True
    else:
        args.eval_metric = 'hits'
        directed = False

    evaluator = Evaluator(name=args.dataset)
    if args.eval_metric == 'hits':
        loggers = {
            'Hits@20': Logger(args.runs, args),
            'Hits@50': Logger(args.runs, args),
            'Hits@100': Logger(args.runs, args),
        }
    elif args.eval_metric == 'mrr':
        loggers = {
            'MRR': Logger(args.runs, args),
        }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    path = dataset.root + '_seal{}'.format(data_appendix)

    if not (args.dynamic_train or args.dynamic_val or args.dynamic_test):
        args.num_workers = 0

    train_dataset, val_dataset, test_dataset = [
        SEALOGBLDataset(path, graph, split_edge, percent=percent, split=split,
            ratio_per_hop=args.ratio_per_hop, directed=directed, dynamic=dynamic)
        for percent, split, dynamic in zip(
            [args.train_percent, args.val_percent, args.test_percent],
            ['train', 'valid', 'test'],
            [args.dynamic_train, args.dynamic_val, args.dynamic_test]
        )
    ]

    max_z = 1000  # set a large max_z so that every z has embeddings to look up
    train_loader = GraphDataLoader(train_dataset, batch_size=args.batch_size, 
                            shuffle=True, collate_fn=ogbl_collate_fn,
                            num_workers=args.num_workers)
    val_loader = GraphDataLoader(val_dataset, batch_size=args.batch_size, 
                            collate_fn=ogbl_collate_fn, num_workers=args.num_workers)
    test_loader = GraphDataLoader(test_dataset, batch_size=args.batch_size, 
                            collate_fn=ogbl_collate_fn, num_workers=args.num_workers)

    for run in range(args.runs):
        model = DGCNN(args.hidden_channels, args.num_layers, max_z, args.sortpool_k, 
            train_dataset, args.dynamic_train, use_feature=args.use_feature).to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f'Total number of parameters is {total_params}')
        print(f'SortPooling k is set to {model.k}')
        with open(log_file, 'a') as f:
            print(f'Total number of parameters is {total_params}', file=f)
            print(f'SortPooling k is set to {model.k}', file=f)

        start_epoch = 1
        # Training starts
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss = train()

            if epoch % args.eval_steps == 0:
                results = test()
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                model_name = os.path.join(
                    args.res_dir, 'run{}_model_checkpoint{}.pth'.format(run+1, epoch))
                optimizer_name = os.path.join(
                    args.res_dir, 'run{}_optimizer_checkpoint{}.pth'.format(run+1, epoch))
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (f'Run: {run + 1:02d}, Epoch: {epoch:02d}, ' +
                                f'Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, ' +
                                f'Test: {100 * test_res:.2f}%')
                    print(key)
                    print(to_print)
                    with open(log_file, 'a') as f:
                        print(key, file=f)
                        print(to_print, file=f)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, 'a') as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, 'a') as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f'Total number of parameters is {total_params}')
    print(f'Results are saved in {args.res_dir}')
