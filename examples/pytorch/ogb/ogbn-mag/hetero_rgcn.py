import argparse
import itertools
from tqdm import tqdm

import dgl
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def extract_embed(node_embed, input_nodes):
    emb = {}
    for ntype, nid in input_nodes.items():
        nid = input_nodes[ntype]
        if ntype in node_embed:
            emb[ntype] = node_embed[ntype][nid]
    return emb

class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph.
    
    Parameters
    ----------
    g : DGLGraph
        Input graph.
    embed_size : int
        The length of each embedding vector
    exclude : list[str]
        The list of node-types to exclude (e.g., because they have natural features)
    """
    def __init__(self, g, embed_size, exclude=list()):
        
        super(RelGraphEmbed, self).__init__()
        self.g = g
        self.embed_size = embed_size

        # create learnable embeddings for all nodes, except those with a node-type in the "exclude" list
        self.embeds = nn.ParameterDict()
        for ntype in g.ntypes:
            if ntype in exclude:
                continue
            embed = nn.Parameter(th.Tensor(g.number_of_nodes(ntype), self.embed_size))
            self.embeds[ntype] = embed

        self.reset_parameters()

    def reset_parameters(self):
        for emb in self.embeds.values():
            nn.init.xavier_uniform_(emb)

    def forward(self, block=None):
        return self.embeds


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    ntypes : list[str]
        Node type names
    rel_names : list[str]
        Relation names.
    weight : bool, optional
        True if a linear layer is applied after message passing. Default: True
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """
    def __init__(self,
                 in_feat,
                 out_feat,
                 ntypes,
                 rel_names,
                 *,
                 weight=True,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.ntypes = ntypes
        self.rel_names = rel_names
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        self.conv = dglnn.HeteroGraphConv({
                rel : dglnn.GraphConv(in_feat, out_feat, norm='right', weight=False, bias=False)
                for rel in rel_names
            })

        self.use_weight = weight
        if self.use_weight:
            self.weight = nn.ModuleDict({
                rel_name: nn.Linear(in_feat, out_feat, bias=False)
                for rel_name in self.rel_names
            })

        # weight for self loop
        if self.self_loop:
            self.loop_weights = nn.ModuleDict({
                ntype: nn.Linear(in_feat, out_feat, bias=bias)
                for ntype in self.ntypes
            })

        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    
    def reset_parameters(self):
        if self.use_weight:
            for layer in self.weight.values():
                layer.reset_parameters()

        if self.self_loop:
            for layer in self.loop_weights.values():
                layer.reset_parameters()

    def forward(self, g, inputs):
        """Forward computation

        Parameters
        ----------
        g : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        g = g.local_var()
        if self.use_weight:
            wdict = {rel_name: {'weight': self.weight[rel_name].weight.T}
                     for rel_name in self.rel_names}
        else:
            wdict = {}

        if g.is_block:
            inputs_dst = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            inputs_dst = inputs

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            if self.self_loop:
                h = h + self.loop_weights[ntype](inputs_dst[ntype])
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)
        return {ntype : _apply(ntype, h) for ntype, h in hs.items()}


class EntityClassify(nn.Module):
    r"""
    R-GCN node classification model

    Parameters
    ----------
    g : DGLGraph
        The heterogenous graph used for message passing
    in_dim : int
        Input feature size.
    h_dim : int
        Hidden dimension size.
    out_dim : int
        Output dimension size.
    num_hidden_layers : int, optional
        Number of RelGraphConvLayers. Default: 1
    dropout : float, optional
        Dropout rate. Default: 0.0
    use_self_loop : bool, optional
        True to include self loop message in RelGraphConvLayers. Default: True
    """
    def __init__(self,
                 g, in_dim,
                 h_dim, out_dim,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=True):
        super(EntityClassify, self).__init__()
        self.g = g
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConvLayer(
            self.in_dim, self.h_dim, g.ntypes, self.rel_names,
            activation=F.relu, self_loop=self.use_self_loop,
            dropout=self.dropout))
        # h2h
        for _ in range(self.num_hidden_layers):
            self.layers.append(RelGraphConvLayer(
                self.h_dim, self.h_dim, g.ntypes, self.rel_names,
                activation=F.relu, self_loop=self.use_self_loop,
                dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConvLayer(
            self.h_dim, self.out_dim, g.ntypes, self.rel_names,
            activation=None,
            self_loop=self.use_self_loop))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


class Logger(object):
    r"""
    This class was taken directly from the PyG implementation and can be found
    here: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/logger.py

    This was done to ensure that performance was measured in precisely the same way
    """
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * th.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f'Run {run + 1:02d}:')
            print(f'Highest Train: {result[:, 0].max():.2f}')
            print(f'Highest Valid: {result[:, 1].max():.2f}')
            print(f'  Final Train: {result[argmax, 0]:.2f}')
            print(f'   Final Test: {result[argmax, 2]:.2f}')
        else:
            result = 100 * th.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = th.tensor(best_results)

            print(f'All runs:')
            r = best_result[:, 0]
            print(f'Highest Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 1]
            print(f'Highest Valid: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 2]
            print(f'  Final Train: {r.mean():.2f} ± {r.std():.2f}')
            r = best_result[:, 3]
            print(f'   Final Test: {r.mean():.2f} ± {r.std():.2f}')


def parse_args():
    # DGL
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.5,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=64,
            help="number of hidden units")
    parser.add_argument("--lr", type=float, default=0.01,
            help="learning rate")
    parser.add_argument("-e", "--n-epochs", type=int, default=3,
            help="number of training epochs")

    # OGB
    parser.add_argument('--runs', type=int, default=10)

    args = parser.parse_args()
    return args

def prepare_data(args):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    split_idx = dataset.get_idx_split()
    g, labels = dataset[0] # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    labels = labels['paper'].flatten()

    def add_reverse_hetero(g, combine_like=True):
        r"""
        Parameters
        ----------
        g : DGLGraph
            The heterogenous graph where reverse edges should be added
        combine_like : bool, optional
            Whether reverse-edges that have identical source/destination 
            node types should be combined with the existing edge-type, 
            rather than creating a new edge type.  Default: True.
        """
        relations = {}
        num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
        for metapath in g.canonical_etypes:
            src_ntype, rel_type, dst_ntype = metapath
            src, dst = g.all_edges(etype=rel_type)

            if src_ntype==dst_ntype and combine_like:
                # Make edges un-directed instead of making a reverse edge type
                relations[metapath] = (th.cat([src, dst], dim=0), th.cat([dst, src], dim=0))
            else:
                # Original edges
                relations[metapath] = (src, dst)

                reverse_metapath = (dst_ntype, 'rev-' + rel_type, src_ntype)
                relations[reverse_metapath] = (dst, src)           # Reverse edges

        new_g = dgl.heterograph(relations, num_nodes_dict=num_nodes_dict)
        # Remove duplicate edges
        new_g = dgl.to_simple(new_g, return_counts=None, writeback_mapping=False, copy_ndata=True)

        # copy_ndata:
        for ntype in g.ntypes:
            for k, v in g.nodes[ntype].data.items():
                new_g.nodes[ntype].data[k] = v.detach().clone()

        return new_g

    g = add_reverse_hetero(g)
    print("Loaded graph: {}".format(g))

    logger = Logger(args['runs'], args)

    # train sampler
    sampler = dgl.dataloading.MultiLayerNeighborSampler(args['fanout'])
    train_loader = dgl.dataloading.DataLoader(
        g, split_idx['train'], sampler,
        batch_size=args['batch_size'], shuffle=True, num_workers=0)
    
    return (g, labels, dataset.num_classes, split_idx,  
            logger, train_loader)

def get_model(g, num_classes, args):
    embed_layer = RelGraphEmbed(g, 128, exclude=['paper'])
    
    model = EntityClassify(
        g, 128, args['n_hidden'], num_classes,
        num_hidden_layers=args['num_layers'] - 2,
        dropout=args['dropout'],
        use_self_loop=True,
    )

    print(embed_layer)
    print(f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}")
    print(model)
    print(f"Number of model parameters: {sum(p.numel() for p in model.parameters())}")

    return embed_layer, model

def train(g, model, node_embed, optimizer, train_loader, split_idx,  
          labels, logger, device, run, args):
    
    # training loop
    print("start training...")
    category = 'paper'

    for epoch in range(args['n_epochs']):
        N_train= split_idx['train'][category].shape[0]
        pbar = tqdm(total=N_train)
        pbar.set_description(f'Epoch {epoch:02d}')
        model.train()
        
        total_loss = 0

        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[category]     # we only predict the nodes with type "category"
            batch_size = seeds.shape[0]

            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features
            emb.update({'paper': g.ndata['feat']['paper'][input_nodes['paper']]})
            lbl = labels[seeds]
            
            if th.cuda.is_available():
                emb = {k : e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            
            optimizer.zero_grad()
            logits = model(emb, blocks)[category]
            
            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_size
            pbar.update(batch_size)
        
        pbar.close()
        loss = total_loss / N_train
        
        result = test(g, model, node_embed, labels, device, split_idx, args)
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(f'Run: {run + 1:02d}, '
              f'Epoch: {epoch +1 :02d}, '
              f'Loss: {loss:.4f}, '
              f'Train: {100 * train_acc:.2f}%, '
              f'Valid: {100 * valid_acc:.2f}%, '
              f'Test: {100 * test_acc:.2f}%')
    
    return logger

@th.no_grad()
def test(g, model, node_embed, y_true, device, split_idx, args):
    model.eval()
    category = 'paper'
    evaluator = Evaluator(name='ogbn-mag')

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args['num_layers'])
    loader = dgl.dataloading.DataLoader(
        g, {'paper': th.arange(g.num_nodes('paper'))}, sampler,
        batch_size=16384, shuffle=False, num_workers=0)
    
    N = y_true.size(0)
    pbar = tqdm(total=N)
    pbar.set_description(f'Full Inference')
    
    y_hats = list()

    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds[category]     # we only predict the nodes with type "category"
        batch_size = seeds.shape[0]

        emb = extract_embed(node_embed, input_nodes)
        # Get the batch's raw "paper" features
        emb.update({'paper': g.ndata['feat']['paper'][input_nodes['paper']]})
        
        if th.cuda.is_available():
            emb = {k : e.cuda() for k, e in emb.items()}
        
        logits = model(emb, blocks)[category]
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())
        
        pbar.update(batch_size)
    
    pbar.close()

    y_pred = th.cat(y_hats, dim=0)
    y_true = th.unsqueeze(y_true, 1)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']['paper']],
        'y_pred': y_pred[split_idx['train']['paper']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']['paper']],
        'y_pred': y_pred[split_idx['valid']['paper']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']['paper']],
        'y_pred': y_pred[split_idx['test']['paper']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main(args):
    # Static parameters
    hyperparameters = dict(
        num_layers=2,
        fanout=[25, 20], 
        batch_size=1024,
    )
    hyperparameters.update(vars(args))
    print(hyperparameters)

    device = f'cuda:0' if th.cuda.is_available() else 'cpu'

    (g, labels, num_classes, split_idx, 
        logger, train_loader) = prepare_data(hyperparameters)

    embed_layer, model = get_model(g, num_classes, hyperparameters)
    model = model.to(device)
    
    for run in range(hyperparameters['runs']):

        embed_layer.reset_parameters()
        model.reset_parameters()

        # optimizer
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=hyperparameters['lr'])

        logger = train(g, model, embed_layer(), optimizer, train_loader, split_idx,
              labels, logger, device, run, hyperparameters)

        logger.print_statistics(run)
    
    print("Final performance: ")
    logger.print_statistics()

if __name__ == '__main__':
    args = parse_args()
    main(args)
