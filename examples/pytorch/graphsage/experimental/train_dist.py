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
from dgl.data.utils import load_graphs
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from pyinstrument import Profiler

from train_sampling import run, SAGE, compute_acc, evaluate, load_subtensor

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

def run(args, device, data):
    # Unpack data
    train_nid, val_nid, in_feats, n_classes, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors)

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
    if not args.standalone:
        model = th.nn.parallel.DistributedDataParallel(model)
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_size = th.sum(g.ndata['train_mask'][0:g.number_of_nodes()])

    # Training loop
    iter_tput = []
    profiler = Profiler()
    profiler.start()
    epoch = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0
        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        step_time = []
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()
            sample_time += tic_step - start

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            start = time.time()
            batch_inputs, batch_labels = load_subtensor(g, seeds, input_nodes, device)
            copy_time += time.time() - start

            num_seeds += len(blocks[-1].dstdata[dgl.NID])
            num_inputs += len(blocks[0].srcdata[dgl.NID])
            # Compute loss and prediction
            start = time.time()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_time += forward_end - start
            backward_time += compute_end - forward_end

            # Aggregate gradients in multiple nodes.
            if not args.standalone:
                for param in model.parameters():
                    if param.requires_grad and param.grad is not None:
                        th.distributed.all_reduce(param.grad.data,
                                                  op=th.distributed.ReduceOp.SUM)
                        param.grad.data /= args.num_client

            optimizer.step()
            update_time += time.time() - compute_end

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(num_seeds / (step_t))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB | time {:.3f} s'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc, np.sum(step_time[-args.log_every:])))
            start = time.time()

        toc = time.time()
        print('Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            toc - tic, sample_time, copy_time, forward_time, backward_time, update_time, num_seeds, num_inputs))
        epoch += 1


        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        #if epoch % args.eval_every == 0 and epoch != 0:
        #    eval_acc = evaluate(model, g, g.ndata['features'], g.ndata['labels'], val_nid, args.batch_size, device)
        #    print('Eval Acc {:.4f}'.format(eval_acc))

    profiler.stop()
    print(profiler.output_text(unicode=True, color=True))
    # clean up
    if not args.standalone:
        g._client.barrier()
        dgl.distributed.shutdown_servers()
        dgl.distributed.finalize_client()

def main(args):
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')
    g = dgl.distributed.DistGraph(args.ip_config, args.graph_name, conf_file=args.conf_path)
    print('rank:', g.rank())

    train_nid = dgl.distributed.node_split(g.ndata['train_mask'], g.get_partition_book(), force_even=True)
    val_nid = dgl.distributed.node_split(g.ndata['val_mask'], g.get_partition_book(), force_even=True)
    test_nid = dgl.distributed.node_split(g.ndata['test_mask'], g.get_partition_book(), force_even=True)
    print('part {}, train: {}, val: {}, test: {}'.format(g.rank(), len(train_nid),
                                                         len(val_nid), len(test_nid)))
    device = th.device('cpu')
    n_classes = len(th.unique(g.ndata['labels'][np.arange(g.number_of_nodes())]))

    # Pack data
    in_feats = g.ndata['features'].shape[1]
    data = train_nid, val_nid, in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph-name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--conf_path', type=str, help='The path to the partition config file')
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
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    args = parser.parse_args()

    print(args)
    main(args)
