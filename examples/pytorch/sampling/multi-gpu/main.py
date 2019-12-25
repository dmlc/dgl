import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import time
import argparse
from dgl.data import RedditDataset

##################################################################################
# GCN using mean reducer
##################################################################################

class GCNNodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(GCNNodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, nodes):
        h = nodes.data['h']
        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return {'h': h} 

class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(GCN, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GCNNodeUpdate(in_feats, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GCNNodeUpdate(n_hidden, n_hidden, activation))
        # output layer
        self.layers.append(
            GCNNodeUpdate(n_hidden, n_classes))

    def forward(self, nf):
        h = nf.layers[0].data['features']
        for i, layer in enumerate(self.layers):
            h = self.dropout(h)
            nf.layers[i].data['h'] = h
            nf.block_compute(i, fn.copy_u('h', 'm'), fn.mean('m', 'h'), layer)
            h = nf.layers[i+1].data['h']
        return h

##################################################################################
# GraphSAGE
##################################################################################

class SAGENodeUpdate(nn.Module):
    def __init__(self, in_feats, out_feats, activation=None):
        super(SAGENodeUpdate, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, nodes):
        h = th.cat([nodes.data['h'], nodes.data['h_n']], dim=1)
        h = self.linear(h)
        if self.activation is not None:
            h = self.activation(h)
        return {'h_new': h}

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 **kwargs):
        super(SAGE, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            SAGENodeUpdate(in_feats * 2, n_hidden, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                SAGENodeUpdate(n_hidden * 2, n_hidden, activation))
        # output layer
        self.layers.append(
            SAGENodeUpdate(n_hidden * 2, n_classes))

    def forward(self, nf):
        for i in range(nf.num_layers):
            nf.layers[i].data['h'] = nf.layers[i].data['features']
            nf.layers[i].data['h_new'] = nf.layers[i].data['features']
        for i in range(len(self.layers)):
            for j in range(i, len(self.layers)):
                nf.layers[j].data['h'] = self.dropout(nf.layers[j].data['h'])
                nf.block_compute(j, fn.copy_u('h', 'm'), fn.mean('m', 'h_n'), self.layers[i])
            for j in range(i, len(self.layers)):
                nf.layers[j+1].data['h'] = nf.layers[j+1].data['h_new']
        return nf.layers[-1].data['h']

##################################################################################
# GAT
##################################################################################

class GATLayer(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 activation=None):
        super(GATLayer, self).__init__()
        self.num_heads = num_heads
        self.out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.activation = activation
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)

    def edge_softmax(self, nf, score, bid):
        nf.blocks[bid].data['s'] = score
        nf.block_compute(bid, fn.copy_e('s', 'm'), fn.max('m', 'smax'))
        nf.apply_block(bid, fn.e_sub_v('s', 'smax', 'out'))
        nf.blocks[bid].data['out'] = th.exp(nf.blocks[bid].data['out'])
        nf.block_compute(bid, fn.copy_e('out', 'm'), fn.sum('m', 'out_sum'))
        nf.apply_block(bid, fn.e_div_v('out', 'out_sum', 'out'))
        return nf.blocks[bid].data['out']

    def forward(self, nf, lid):
        l_ft = nf.layers[lid].data['h'].flatten(1)
        l_ft = self.fc(l_ft).view(-1, self.num_heads, self.out_feats)
        el = (l_ft * self.attn_l).sum(dim=-1).unsqueeze(-1)
        nf.layers[lid].data.update({'ft': l_ft, 'el': el})

        r_ft = nf.layers[lid+1].data['h'].flatten(1)
        r_ft = self.fc(r_ft).view(-1, self.num_heads, self.out_feats)
        er = (r_ft * self.attn_r).sum(dim=-1).unsqueeze(-1)
        nf.layers[lid+1].data.update({'er' : er})

        # compute edge attention
        nf.apply_block(lid, fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(nf.blocks[lid].data.pop('e'))
        # compute softmax
        nf.blocks[lid].data['a'] = self.edge_softmax(nf, e, lid)

        # message passing
        nf.block_compute(lid, fn.u_mul_e('ft', 'a', 'm'), fn.sum('m', 'ft'))
        rst = nf.layers[lid+1].data['ft']

        if self.activation:
            rst = self.activation(rst)
        return rst

class GAT(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 num_heads,
                 activation,
                 dropout,
                 **kwargs):
        super(GAT, self).__init__(**kwargs)
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(
            GATLayer(in_feats, n_hidden, num_heads, activation))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(
                GATLayer(n_hidden * num_heads, n_hidden, num_heads, activation))
        # output layer
        self.layers.append(
            GATLayer(n_hidden * num_heads, n_classes, num_heads))

    def forward(self, nf):
        for i in range(nf.num_layers):
            nf.layers[i].data['h'] = nf.layers[i].data['features']
            nf.layers[i].data['h_new'] = nf.layers[i].data['features']
        for i in range(len(self.layers)):
            for j in range(i, len(self.layers)):
                nf.layers[j].data['h'] = self.dropout(nf.layers[j].data['h'])
            for j in range(i, len(self.layers)):
                h_new = self.layers[i](nf, j)
                nf.layers[j+1].data['h_new'] = h_new
            for j in range(i, len(self.layers)):
                nf.layers[j+1].data['h'] = nf.layers[j+1].data['h_new']
        h = nf.layers[-1].data['h']
        return h.mean(1)

def compute_acc(pred, labels):
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, nf, labels):
    model.eval()
    with th.no_grad():
        pred = model(nf)
    model.train()
    return compute_acc(pred, labels)

def run(proc_id, n_gpus, args, devices):
    #th.manual_seed(1234)
    #np.random.seed(1234)
    #th.cuda.manual_seed_all(1234)
    #dgl.random.seed(1234)

    # dropout probability
    dropout = 0.2

    # Setup multi-process
    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.set_num_threads(args.num_workers * 2 if args.prefetch else args.num_workers)

    # Prepare data
    data = RedditDataset(self_loop=True)
    train_nid = th.LongTensor(np.nonzero(data.train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(data.val_mask)[0])
    features = th.Tensor(data.features)
    # Split train_nid
    train_nid = th.split(train_nid, len(train_nid) // n_gpus)[dev_id]

    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels

    # Construct graph
    g = dgl.DGLGraph(data.graph, readonly=True)
    g.ndata['features'] = features

    # Create sampler
    sampler = dgl.contrib.sampling.NeighborSampler(
        g, args.batch_size, args.fan_out,
        neighbor_type='in',
        shuffle=True,
        num_hops=args.num_layers,
        seed_nodes=train_nid,
        num_workers=args.num_workers)

    if proc_id == 0:
        val_sampler = dgl.contrib.sampling.NeighborSampler(
            g, len(val_nid), 10000,
            neighbor_type='in',
            shuffle=False,
            num_hops=args.num_layers,
            seed_nodes=val_nid,
            num_workers=1)
        # Create validation batch (only on GPU 0)
        val_nf = list(val_sampler)[0]
        val_nf.copy_from_parent()
        for i in range(val_nf.num_layers):
            val_nf.layers[i].data['features'] = val_nf.layers[i].data['features'].to(0)

    # Define model and optimizer
    if args.model == 'gcn':
        model = GCN(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)
    elif args.model == 'sage':
        model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)
    elif args.model == 'gat':
        model = GAT(in_feats, args.num_hidden, n_classes, args.num_layers, 4, F.relu, dropout)
    else:
        raise ValueError('Unknown model name:', args.model)
    model = model.to(dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()
        for step, nf in enumerate(sampler):
            if proc_id == 0:
                tic_step = time.time()
            nf.copy_from_parent()
            for i in range(nf.num_layers):
                nf.layers[i].data['features'] = nf.layers[i].data['features'].to(dev_id)
            # forward
            pred = model(nf)
            batch_nids = nf.layer_parent_nid(-1).to(dev_id)
            batch_labels = labels[batch_nids].to(dev_id)
            # compute loss
            loss = loss_fcn(pred, batch_labels)
            # backward
            optimizer.zero_grad()
            loss.backward()
            if n_gpus > 1:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data,
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= n_gpus
            #if n_gpus > 1:
                #th.distributed.barrier()
            optimizer.step()
            if proc_id == 0:
                iter_tput.append(len(batch_nids) * n_gpus / (time.time() - tic_step))
            if step % args.log_every == 0 and proc_id == 0:
                acc = compute_acc(pred, batch_labels)
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f}'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:])))
        if n_gpus > 1:
            th.distributed.barrier()
        toc = time.time()
        if proc_id == 0:
            print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if epoch % args.eval_every == 0 and epoch != 0:
                eval_acc = evaluate(model, val_nf, labels[val_nid].to(dev_id))
                print('Eval Acc {:.4f}'.format(eval_acc))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--model', type=str, default='gcn')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-workers', type=int, default=1)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--prefetch', action='store_true')
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.03)
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)
    if n_gpus == 1:
        run(0, n_gpus, args, devices)
    else:
        mp = th.multiprocessing
        mp.spawn(run, args=(n_gpus, args, devices), nprocs=n_gpus)
