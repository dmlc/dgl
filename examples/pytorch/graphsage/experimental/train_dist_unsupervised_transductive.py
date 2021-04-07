import os
os.environ['DGLBACKEND']='pytorch'
from multiprocessing import Process
import argparse, time, math
import numpy as np
from functools import wraps
import tqdm
import sklearn.linear_model as lm
import sklearn.metrics as skm

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
from dgl.distributed import DistDataLoader

from pyinstrument import Profiler
from dgl.distributed import NodeEmbedding
from dgl.distributed.optim import SparseAdagrad
from train_dist_unsupervised import SAGE, NeighborSampler, PosNeighborSampler, CrossEntropyLoss, compute_acc

def initializer(shape, dtype):
    arr = th.zeros(shape, dtype=dtype)
    arr.uniform_(-1, 1)
    return arr

class DistEmb(nn.Module):
    def __init__(self, num_nodes, emb_size, dev_id='cpu'):
        super().__init__()
        self.dev_id = dev_id
        self.emb = NodeEmbedding(num_nodes, emb_size, name='sage', init_func=initializer)

    def forward(self, idx):
        return self.emb(idx, device=self.dev_id)

class DistSAGE(SAGE):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super(DistSAGE, self).__init__(in_feats, n_hidden, n_classes, n_layers,
                                       activation, dropout)

    def inference(self, g, emb_layer, batch_size, device):
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
        nodes = dgl.distributed.node_split(np.arange(g.number_of_nodes()),
                                           g.get_partition_book(), force_even=True)
        y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_hidden), th.float32, 'h',
                                       persistent=True)
        for l, layer in enumerate(self.layers):
            if l == len(self.layers) - 1:
                y = dgl.distributed.DistTensor((g.number_of_nodes(), self.n_classes),
                                               th.float32, 'h_last', persistent=True)

            sampler = PosNeighborSampler(g, [-1], dgl.distributed.sample_neighbors)
            print('|V|={}, eval batch size: {}'.format(g.number_of_nodes(), batch_size))
            # Create PyTorch DataLoader for constructing blocks
            dataloader = DistDataLoader(
                dataset=nodes,
                batch_size=batch_size,
                collate_fn=sampler.sample_blocks,
                shuffle=False,
                drop_last=False)

            for blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                input_nodes = block.srcdata[dgl.NID]
                output_nodes = block.dstdata[dgl.NID]
                if l == 0:
                    h = emb_layer(input_nodes)
                else:
                    h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata['features'][input_nodes].to(device)
    return batch_inputs

def generate_emb(model, emb_layer, g, batch_size, device):
    """
    Generate embeddings for each node
    g : The entire graph.
    inputs : The features of all the nodes.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        pred = model.inference(g, emb_layer, batch_size, device)

    return pred

def run(args, device, data):
    # Unpack data
    train_eids, train_nids, g, global_train_nid, global_valid_nid, global_test_nid, labels = data
    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], train_nids,
                              dgl.distributed.sample_neighbors, args.num_negs, args.remove_edge)

    # Create PyTorch DataLoader for constructing blocks
    dataloader = dgl.distributed.DistDataLoader(
        dataset=train_eids.numpy(),
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False)

    # Define model and optimizer
    emb_layer = DistEmb(g.num_nodes(), args.num_hidden, device)
    model = DistSAGE(args.num_hidden, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    sparse_optimizer = SparseAdagrad([emb_layer.emb], lr=args.lr)

    # Training loop
    epoch = 0
    #profiler = Profiler()
    #if g.rank() == 0:
    #    profiler.start()
    for epoch in range(args.num_epochs):
        sample_time = 0
        copy_time = 0
        forward_time = 0
        backward_time = 0
        update_time = 0
        num_seeds = 0
        num_inputs = 0

        step_time = []
        iter_t = []
        sample_t = []
        feat_copy_t = []
        forward_t = []
        backward_t = []
        update_t = []
        iter_tput = []

        start = time.time()
        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        for step, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            tic_step = time.time()
            sample_t.append(tic_step - start)

            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.to(device) for block in blocks]
            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.

            # Load the input features as well as output labels
            batch_inputs = blocks[0].srcdata[dgl.NID]
            copy_time = time.time()
            feat_copy_t.append(copy_time - tic_step)

            # Compute loss and prediction
            batch_inputs = emb_layer(batch_inputs)
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            forward_end = time.time()
            sparse_optimizer.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_t.append(forward_end - copy_time)
            backward_t.append(compute_end - forward_end)

            # Aggregate gradients in multiple nodes.
            sparse_optimizer.step()
            optimizer.step()
            update_t.append(time.time() - compute_end)

            pos_edges = pos_graph.number_of_edges()
            neg_edges = neg_graph.number_of_edges()

            step_t = time.time() - start
            step_time.append(step_t)
            iter_tput.append(pos_edges / step_t)
            num_seeds += pos_edges
            if step % args.log_every == 0:
                print('[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f} | time {:.3f} s' \
                        '| sample {:.3f} | copy {:.3f} | forward {:.3f} | backward {:.3f} | update {:.3f}'.format(
                    g.rank(), epoch, step, loss.item(), np.mean(iter_tput[3:]), np.sum(step_time[-args.log_every:]),
                    np.sum(sample_t[-args.log_every:]), np.sum(feat_copy_t[-args.log_every:]), np.sum(forward_t[-args.log_every:]),
                    np.sum(backward_t[-args.log_every:]), np.sum(update_t[-args.log_every:])))

                #if g.rank() == 0:
                #    profiler.stop()
                #    print(profiler.output_text(unicode=True, color=True, show_all=True))
                #    profiler.start()

            start = time.time()

        print('[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            g.rank(), np.sum(step_time), np.sum(sample_t), np.sum(feat_copy_t), np.sum(forward_t), np.sum(backward_t), np.sum(update_t), num_seeds, num_inputs))
        epoch += 1

    # evaluate the embedding using LogisticRegression
    if args.standalone:
        pred = generate_emb(model, emb_layer, g, args.batch_size_eval, device)
    else:
        pred = generate_emb(model.module, emb_layer, g, args.batch_size_eval, device)
    if g.rank() == 0:
        eval_acc, test_acc = compute_acc(pred, labels, global_train_nid, global_valid_nid, global_test_nid)
        print('eval acc {:.4f}; test acc {:.4f}'.format(eval_acc, test_acc))

    # sync for eval and test
    if not args.standalone:
        th.distributed.barrier()

    if not args.standalone:
        g._client.barrier()

        # save features into file
        if g.rank() == 0:
            th.save(pred, 'emb.pt')
    else:
        feat = g.ndata['features']
        th.save(pred, 'emb.pt')

def main(args):
    dgl.distributed.initialize(args.ip_config, args.num_servers, num_workers=args.num_workers)
    if not args.standalone:
        th.distributed.init_process_group(backend='gloo')
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print('rank:', g.rank())
    print('number of edges', g.number_of_edges())

    train_eids = dgl.distributed.edge_split(th.ones((g.number_of_edges(),), dtype=th.bool), g.get_partition_book(), force_even=True)
    train_nids = dgl.distributed.node_split(th.ones((g.number_of_nodes(),), dtype=th.bool), g.get_partition_book())
    global_train_nid = th.LongTensor(np.nonzero(g.ndata['train_mask'][np.arange(g.number_of_nodes())]))
    global_valid_nid = th.LongTensor(np.nonzero(g.ndata['val_mask'][np.arange(g.number_of_nodes())]))
    global_test_nid = th.LongTensor(np.nonzero(g.ndata['test_mask'][np.arange(g.number_of_nodes())]))
    labels = g.ndata['labels'][np.arange(g.number_of_nodes())]
    if args.num_gpus == -1:
        device = th.device('cpu')
    else:
        device = th.device('cuda:'+str(g.rank() % args.num_gpus))

    # Pack data
    global_train_nid = global_train_nid.squeeze()
    global_valid_nid = global_valid_nid.squeeze()
    global_test_nid = global_test_nid.squeeze()
    print("number of train {}".format(global_train_nid.shape[0]))
    print("number of valid {}".format(global_valid_nid.shape[0]))
    print("number of test {}".format(global_test_nid.shape[0]))
    data = train_eids, train_nids, g, global_train_nid, global_valid_nid, global_test_nid, labels
    run(args, device, data)
    print("parent ends")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', type=str, help='graph name')
    parser.add_argument('--id', type=int, help='the partition id')
    parser.add_argument('--ip_config', type=str, help='The file for IP configuration')
    parser.add_argument('--part_config', type=str, help='The path to the partition config file')
    parser.add_argument('--num_servers', type=int, default=1, help='Server count on each machine.')
    parser.add_argument('--n_classes', type=int, help='the number of classes')
    parser.add_argument('--num_gpus', type=int, default=-1,
                        help="the number of GPU device. Use -1 for CPU training")
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--num_hidden', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--fan_out', type=str, default='10,25')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--batch_size_eval', type=int, default=100000)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    parser.add_argument('--local_rank', type=int, help='get rank of the process')
    parser.add_argument('--standalone', action='store_true', help='run in the standalone mode')
    parser.add_argument('--num_negs', type=int, default=1)
    parser.add_argument('--neg_share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    parser.add_argument('--remove_edge', default=False, action='store_true',
        help="whether to remove edges during sampling")
    args = parser.parse_args()
    assert args.num_workers == int(os.environ.get('DGL_NUM_SAMPLER')), \
    'The num_workers should be the same value with DGL_NUM_SAMPLER.'
    assert args.num_servers == int(os.environ.get('DGL_NUM_SERVER')), \
    'The num_servers should be the same value with DGL_NUM_SERVER.'

    print(args)
    main(args)
