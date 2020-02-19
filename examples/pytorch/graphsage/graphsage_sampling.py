import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel

class SAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(
            in_feats, n_hidden, 'mean', feat_drop=dropout, activation=activation))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(
                n_hidden, n_hidden, 'mean', feat_drop=dropout, activation=activation))
        self.layers.append(dglnn.SAGEConv(
            n_hidden, n_classes, 'mean', feat_drop=dropout, activation=activation))

    def forward(self, frontiers, x):
        h = x
        for layer, frontier in zip(self.layers, frontiers):
            h = layer(frontier, h)
        return h

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_frontiers(self, seeds):
        frontiers = []
        for fanout in self.fanouts:
            frontier = dgl.sampling.sample_neighbors(g, seeds, fanout)
            seeds = th.unique(th.cat(frontier.all_edges()))
            frontiers.insert(0, frontier)
        return dgl.compact_graphs(frontiers)

def compute_acc(pred, labels):
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, frontiers, inputs, labels, mask):
    model.eval()
    with th.no_grad():
        pred = model(frontiers, inputs)
    model.train()
    return compute_acc(pred[mask], labels[mask])

def run(proc_id, n_gpus, args, devices, data):
    dropout = 0.2

    dev_id = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)
    th.cuda.set_device(dev_id)

    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Split train_nid
    train_nid = th.split(train_nid, len(train_nid) // n_gpus)[dev_id]

    # Create sampler
    sampler = NeighborSampler(g, [args.fan_out] * args.num_layers)
    val_frontiers = sampler.sample_frontiers(val_nid)
    val_induced_nodes = val_frontiers[0].ndata[dgl.NID]
    batch_val_inputs = g.ndata['features'][val_induced_nodes]
    batch_val_mask = val_mask[val_induced_nodes]
    batch_val_labels = labels[val_induced_nodes]

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, dropout)
    model = model.to(dev_id)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(dev_id)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()
        train_nid_batches = train_nid[th.randperm(len(train_nid))].split(args.batch_size)
        for step, seeds in enumerate(train_nid_batches):
            if proc_id == 0:
                tic_step = time.time()

            frontiers = sampler.sample_frontiers(seeds)
            induced_nodes = frontiers.ndata[dgl.NID]
            batch_inputs = g.ndata['features'][induced_nodes].to(dev_id)
            batch_train_mask = train_mask[induced_nodes].to(dev_id)
            batch_labels = labels[induced_nodes].to(dev_id)

            # forward
            batch_pred = model(frontiers, batch_inputs)[batch_train_mask]
            batch_labels = batch_labels[batch_train_mask]
            # compute loss
            loss = loss_fcn(batch_pred, batch_labels)
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
                acc = compute_acc(batch_pred, batch_labels)
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
                eval_acc = evaluate(model, val_frontiers, batch_val_inputs, batch_val_labels, batch_val_mask)
                print('Eval Acc {:.4f}'.format(eval_acc))

    if n_gpus > 1:
        th.distributed.barrier()
    if proc_id == 0:
        print('Avg epoch time: {}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=str, default='0')
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=64)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=int, default=10)
    argparser.add_argument('--batch-size', type=int, default=32)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.03)
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    features = th.Tensor(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        mp = th.multiprocessing
        for proc_id in range(n_gpus):
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
