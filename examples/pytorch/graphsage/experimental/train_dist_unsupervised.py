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
#from pyinstrument import Profiler

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
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
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
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.sampling.MultiLayerNeighborSampler([None])
            dataloader = dgl.sampling.NodeDataLoader(
                g,
                th.arange(g.number_of_nodes()),
                sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


class NegativeSampler(object):
    def __init__(self, g, neg_nseeds):
        self.neg_nseeds = neg_nseeds

    def __call__(self, num_samples):
        # select local neg nodes as seeds
        return self.neg_nseeds[th.randint(self.neg_nseeds.shape[0], (num_samples,))]

class NeighborSampler(object):
    def __init__(self, g, fanouts, neg_nseeds, sample_neighbors, num_negs, remove_edge):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.neg_sampler = NegativeSampler(g, neg_nseeds)
        self.num_negs = num_negs
        self.remove_edge = remove_edge

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
            if self.remove_edge:
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

        input_nodes = blocks[0].srcdata[dgl.NID]
        blocks[0].srcdata['features'] = load_subtensor(self.g, input_nodes, 'cpu')
        # Pre-generate CSR format that it can be used in training directly
        return pos_graph, neg_graph, blocks

class PosNeighborSampler(object):
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

class DistSAGE(SAGE):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers,
                 activation, dropout):
        super(DistSAGE, self).__init__(in_feats, n_hidden, n_classes, n_layers,
                                       activation, dropout)

    def inference(self, g, x, batch_size, device):
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

def generate_emb(model, g, inputs, batch_size, device):
    """
    Generate embeddings for each node
    g : The entire graph.
    inputs : The features of all the nodes.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)

    return pred

def compute_acc(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    
    We will fist train a LogisticRegression model using the trained embeddings,
    the training set, validation set and test set is provided as the arguments.

    The final result is predicted by the lr model.

    emb: The pretrained embeddings
    labels: The ground truth
    train_nids: The training set node ids
    val_nids: The validation set node ids
    test_nids: The test set node ids
    """

    emb = emb[np.arange(labels.shape[0])].cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    val_nids = val_nids.cpu().numpy()
    test_nids = test_nids.cpu().numpy()
    labels = labels.cpu().numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], labels[train_nids])

    pred = lr.predict(emb)
    eval_acc = skm.accuracy_score(labels[val_nids], pred[val_nids])
    test_acc = skm.accuracy_score(labels[test_nids], pred[test_nids])
    return eval_acc, test_acc

def run(args, device, data):
    # Unpack data
    train_eids, train_nids, in_feats, g, global_train_nid, global_valid_nid, global_test_nid, labels = data
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
    model = DistSAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
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

    # Training loop
    #profiler = Profiler()
    #profiler.start()
    epoch = 0
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
            batch_inputs = blocks[0].srcdata['features']
            copy_time = time.time()
            feat_copy_t.append(copy_time - tic_step)

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            forward_end = time.time()
            optimizer.zero_grad()
            loss.backward()
            compute_end = time.time()
            forward_t.append(forward_end - copy_time)
            backward_t.append(compute_end - forward_end)

            # Aggregate gradients in multiple nodes.
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
            start = time.time()

        print('[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, #inputs: {}'.format(
            g.rank(), np.sum(step_time), np.sum(sample_t), np.sum(feat_copy_t), np.sum(forward_t), np.sum(backward_t), np.sum(update_t), num_seeds, num_inputs))
        epoch += 1

    # evaluate the embedding using LogisticRegression
    if args.standalone:
        pred = generate_emb(model,g, g.ndata['features'], args.batch_size_eval, device)
    else:
        pred = generate_emb(model.module, g, g.ndata['features'], args.batch_size_eval, device)
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
    in_feats = g.ndata['features'].shape[1]
    global_train_nid = global_train_nid.squeeze()
    global_valid_nid = global_valid_nid.squeeze()
    global_test_nid = global_test_nid.squeeze()
    print("number of train {}".format(global_train_nid.shape[0]))
    print("number of valid {}".format(global_valid_nid.shape[0]))
    print("number of test {}".format(global_test_nid.shape[0]))
    data = train_eids, train_nids, in_feats, g, global_train_nid, global_valid_nid, global_test_nid, labels
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
