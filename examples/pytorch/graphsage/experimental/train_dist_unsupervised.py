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

class NegativeSampler(object):
    def __init__(self, g, neg_nseeds):
        self.neg_nseeds = neg_nseeds

    def __call__(self, num_samples):
        # select local neg nodes as seeds
        return self.neg_nseeds[th.randint(self.neg_nseeds.shape[0], (num_samples,))]

class NeighborSampler(object):
    def __init__(self, g, fanouts, neg_nseeds, sample_neighbors, num_negs):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.neg_sampler = NegativeSampler(g, neg_nseeds)
        self.num_negs = num_negs

    def sample_blocks(self, seed_edges):
        n_edges = len(seed_edges)
        seed_edges = th.LongTensor(np.asarray(seed_edges))
        heads, tails = self.g.find_edges(seed_edges)

        neg_tails = self.neg_sampler(self.num_negs * n_edges)
        neg_heads = heads.view(-1, 1).expand(n_edges, self.num_negs).flatten()

        # Maintain the correspondence between heads, tails and negative tails as two
        # graphs.
        # pos_graph contains the correspondence between each head and its positive tail.
        # neg_graph contains the correspondence between each head and its negative tails.
        # Both pos_graph and neg_graph are first constructed with the same node space as
        # the original graph.  Then they are compacted together with dgl.compact_graphs.
        pos_graph = dgl.graph((heads, tails), num_nodes=self.g.number_of_nodes())
        neg_graph = dgl.graph((neg_heads, neg_tails), num_nodes=self.g.number_of_nodes())
        pos_graph, neg_graph = dgl.compact_graphs([pos_graph, neg_graph])

        seeds = pos_graph.ndata[dgl.NID]
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Remove all edges between heads and tails, as well as heads and neg_tails.
            _, _, edge_ids = frontier.edge_ids(
                th.cat([heads, tails, neg_heads, neg_tails]),
                th.cat([tails, heads, neg_tails, neg_heads]),
                return_uv=True)
            frontier = dgl.remove_edges(frontier, edge_ids)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)

            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)

        # Pre-generate CSR format that it can be used in training directly
        return pos_graph, neg_graph, blocks

def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    return batch_inputs

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
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from the
            # appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of h_dst
            # would be (num_nodes_RHS, D)
            h_dst = h[:block.number_of_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def run(args, device, data):
    # Unpack data
    train_eids, train_nids, in_feats, g = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], train_nids,
                              dgl.distributed.sample_neighbors, args.num_negs)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = DataLoader(
        dataset=train_eids.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        model = th.nn.parallel.DistributedDataParallel(model)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    iter_tput = []
    sample_tput = []
    profiler = Profiler()
    #profiler.start()
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
        for step, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            tic_step = time.time()
            sample_time = tic_step - start

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]

            # Load the input features as well as output labels
            start = time.time()
            batch_inputs = load_subtensor(g, input_nodes, device)
            copy_time += time.time() - start

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
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

            pos_edges = pos_graph.number_of_edges()
            neg_edges = neg_graph.number_of_edges()

            step_t = time.time() - tic_step
            step_time.append(step_t)
            iter_tput.append(pos_edges / (step_t + sample_time))
            sample_tput.append(sample_time)
            if step % args.log_every == 0:
                print('[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | time {:.3f}/{: .3f} s'.format(
                    g.rank(), epoch, step, loss.item(), np.mean(iter_tput[3:]), np.sum(step_time[-args.log_every:]), np.sum(sample_tput[-args.log_every:]))
            start = time.time()

        toc = time.time()
        print('[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            g.rank(), toc - tic, sample_time, copy_time, forward_time, backward_time, update_time, num_seeds, num_inputs))
        epoch += 1

        #if epoch % args.eval_every == 0 and epoch != 0:
        #    eval_acc = evaluate(model, g, g.ndata['features'], g.ndata['labels'], val_nid, args.batch_size, device)
        #    print('Eval Acc {:.4f}'.format(eval_acc))

    #profiler.stop()
    #print(profiler.output_text(unicode=True, color=True))
    # clean up
    if not args.standalone:
        g._client.barrier()

        if g.rank() == 0:
            feat = g.ndata['features']
            th.save(feat, 'feat.pt')

        dgl.distributed.shutdown_servers()
        dgl.distributed.finalize_client()
    else:
        feat = g.ndata['features']
        th.save(feat, 'feat.pt')

def main(args):
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')
    g = dgl.distributed.DistGraph(args.ip_config, args.graph_name, conf_file=args.conf_path)
    print('rank:', g.rank())
    print('number of edges', g.number_of_edges())

    train_eids = dgl.distributed.edge_split(th.arange(g.number_of_edges()), g.get_partition_book(), force_even=True)
    train_nids = dgl.distributed.node_split(th.arange(g.number_of_nodes()), g.get_partition_book())
    device = th.device('cpu')

    # Pack data
    in_feats = g.ndata['features'].shape[1]
    data = train_eids, train_nids, in_feats, g
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
    parser.add_argument('--num-negs', type=int, default=1)
    parser.add_argument('--neg-share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    args = parser.parse_args()

    print(args)
    main(args)
