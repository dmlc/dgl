import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm

import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.contrib import DistGraphStoreServer, DistGraphStore
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from train_sampling import run, NeighborSampler, SAGE, compute_acc, evaluate

def start_server(args):
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)
    serv = DistGraphStoreServer(server_namebook, args.id, args.graph_name,
                                args.server_data, args.client_data, args.num_client)
    serv.start()

def copy_from_kvstore(nfs, g, stats):
    num_bytes = 0
    num_local_bytes = 0
    first_layer_nid = []
    last_layer_nid = []
    first_layer_offs = [0]
    last_layer_offs = [0]
    for nf in nfs:
        first_layer_nid.append(nf.layer_parent_nid(0))
        last_layer_nid.append(nf.layer_parent_nid(-1))
        first_layer_offs.append(first_layer_offs[-1] + len(nf.layer_parent_nid(0)))
        last_layer_offs.append(last_layer_offs[-1] + len(nf.layer_parent_nid(-1)))
    first_layer_nid = torch.cat(first_layer_nid, dim=0)
    last_layer_nid = torch.cat(last_layer_nid, dim=0)

    # TODO we need to gracefully handle the case that the nodes don't exist.
    start = time.time()
    first_layer_data = g.get_ndata('features', first_layer_nid)
    last_layer_data = g.get_ndata('labels', last_layer_nid)
    stats[2] = time.time() - start
    first_layer_local = g.is_local(first_layer_nid).numpy()
    last_layer_local = g.is_local(last_layer_nid).numpy()
    num_bytes += np.prod(first_layer_data.shape)
    num_bytes += np.prod(last_layer_data.shape)
    if len(first_layer_data.shape) == 1:
        num_local_bytes += np.sum(first_layer_local)
    else:
        num_local_bytes += np.sum(first_layer_local) * first_layer_data.shape[1]
    if len(last_layer_data.shape) == 1:
        num_local_bytes += np.sum(last_layer_local)
    else:
        num_local_bytes += np.sum(last_layer_local) * last_layer_data.shape[1]

    for idx, nf in enumerate(nfs):
        start = first_layer_offs[idx]
        end = first_layer_offs[idx + 1]
        nfs[idx]._node_frames[0]['features'] = first_layer_data[start:end]
        start = last_layer_offs[idx]
        end = last_layer_offs[idx + 1]
        nfs[idx]._node_frames[-1]['labels'] = last_layer_data[start:end]

    stats[0] = num_bytes * 4
    stats[1] = num_local_bytes * 4

def load_subtensor(g, labels, seeds, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    
    batch_inputs = g.get_ndata('features', input_nodes)
    batch_labels = g.get_ndata('labels', seeds).long()
    return batch_inputs, batch_labels

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, in_feats, labels, n_classes, g = data
    # Create sampler
    sampler = NeighborSampler(g.g, [int(fanout) for fanout in args.fan_out.split(',')])

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_nid.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_tput = []
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(len(seeds) / (step_t))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | time {:.3f} s'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc, np.sum(step_time[-args.log_every:])))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['features'], labels, val_nid, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))

    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

def main(args):
    server_namebook = dgl.contrib.read_ip_config(filename=args.ip_config)

    g = DistGraphStore(server_namebook, args.graph_name)
    hg = dgl.graph(g.g.all_edges())
    hg.ndata[dgl.NID] = g.g.ndata[dgl.NID]
    hg.edata[dgl.EID] = g.g.edata[dgl.EID]
    g.g = hg

    # We need to set random seed here. Otherwise, all processes have the same mini-batches.
    th.manual_seed(g.get_id())
    train_mask = g.get_ndata('train_mask').numpy()
    val_mask = g.get_ndata('val_mask').numpy()
    test_mask = g.get_ndata('test_mask').numpy()
    print('part {}, train: {}, val: {}, test: {}'.format(g.get_id(),
        np.sum(train_mask), np.sum(val_mask), np.sum(test_mask)), flush=True)

    train_nid = th.tensor(g.get_local_nids()[train_mask == 1], dtype=th.int64)
    val_nid = th.tensor(g.get_local_nids()[val_mask == 1], dtype=th.int64)
    test_nid = th.tensor(g.get_local_nids()[test_mask == 1], dtype=th.int64)
    labels = g.get_ndata('labels').long()
    
    device = th.device('cpu')

    # Pack data
    data = train_nid, val_nid, args.n_features, labels, args.n_classes, g
    if args.model == "gcn_ns":
        run(args, device, data)
    else:
        print("unknown model. Please choose from gcn_ns, gcn_cv, graphsage_cv")
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument("--model", type=str,
                        help="select a model. Valid models: gcn_ns, gcn_cv, graphsage_cv")
    parser.add_argument('--server', action='store_true',
            help='whether this is a server.')
    parser.add_argument('--graph-name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--n-features', type=int, help='the input feature size')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--server_data', type=str, help='The file with the server data')
    parser.add_argument('--client_data', type=str, help='The file with data exposed to the client.')
    parser.add_argument('--num-client', type=int, help='The number of clients')
    parser.add_argument('--n-classes', type=int, help='the number of classes')
    parser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    parser.add_argument('--num-epochs', type=int, default=20)
    parser.add_argument('--num-hidden', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fan-out', type=str, default='10,25')
    parser.add_argument('--batch-size', type=int, default=1000)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = parser.parse_args()

    print(args)

    if args.server:
        start_server(args)
    else:
        main(args)
