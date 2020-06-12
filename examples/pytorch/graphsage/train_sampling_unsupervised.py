import dgl
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
import dgl.function as fn
import dgl.nn.pytorch as dglnn
import time
import argparse
from _thread import start_new_thread
from functools import wraps
from dgl.data import RedditDataset
from torch.nn.parallel import DistributedDataParallel
import tqdm
import traceback
import sklearn.linear_model as lm
import sklearn.metrics as skm

from utils import thread_wrapped_func

#### Negative sampler

class NegativeSampler(object):
    def __init__(self, g):
        self.weights = g.in_degrees().float() ** 0.75

    def __call__(self, num_samples):
        return self.weights.multinomial(num_samples, replacement=True)

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts, num_negs, neg_share=False):
        self.g = g
        self.fanouts = fanouts
        self.neg_sampler = NegativeSampler(g)
        self.num_negs = num_negs
        self.neg_share = neg_share

    def sample_blocks(self, seed_edges):
        n_edges = len(seed_edges)
        seed_edges = th.LongTensor(np.asarray(seed_edges))
        heads, tails = self.g.find_edges(seed_edges)
        if self.neg_share and n_edges % self.num_negs == 0:
            neg_tails = self.neg_sampler(n_edges)
            neg_tails = neg_tails.view(-1, 1, self.num_negs).expand(n_edges//self.num_negs,
                                                                    self.num_negs,
                                                                    self.num_negs).flatten()
            neg_heads = heads.view(-1, 1).expand(n_edges, self.num_negs).flatten()
        else:
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

        # Obtain the node IDs being used in either pos_graph or neg_graph.  Since they
        # are compacted together, pos_graph and neg_graph share the same compacted node
        # space.
        seeds = pos_graph.ndata[dgl.NID]
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(self.g, seeds, fanout, replace=True)
            # Remove all edges between heads and tails, as well as heads and neg_tails.
            _, _, edge_ids = frontier.edge_ids(
                th.cat([heads, tails, neg_heads, neg_tails]),
                th.cat([tails, heads, neg_tails, neg_heads]),
                return_uv=True)
            frontier = dgl.remove_edges(frontier, edge_ids)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)

            # Pre-generate CSR format that it can be used in training directly
            block.in_degree(0)
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
        nodes = th.arange(g.number_of_nodes())
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.number_of_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            for start in tqdm.trange(0, len(nodes), batch_size):
                end = start + batch_size
                batch_nodes = nodes[start:end]
                block = dgl.to_block(dgl.in_subgraph(g, batch_nodes), batch_nodes)
                input_nodes = block.srcdata[dgl.NID]

                h = x[input_nodes].to(device)
                h_dst = h[:block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

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

def compute_acc(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.
    """
    emb = emb.cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    train_labels = labels[train_nids].cpu().numpy()
    val_nids = val_nids.cpu().numpy()
    val_labels = labels[val_nids].cpu().numpy()
    test_nids = test_nids.cpu().numpy()
    test_labels = labels[test_nids].cpu().numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)

    lr = lm.LogisticRegression(multi_class='multinomial', max_iter=10000)
    lr.fit(emb[train_nids], labels[train_nids])

    pred = lr.predict(emb)
    f1_micro_eval = skm.f1_score(labels[val_nids], pred[val_nids], average='micro')
    f1_micro_test = skm.f1_score(labels[test_nids], pred[test_nids], average='micro')
    f1_macro_eval = skm.f1_score(labels[val_nids], pred[val_nids], average='macro')
    f1_macro_test = skm.f1_score(labels[test_nids], pred[test_nids], average='macro')
    return f1_micro_eval, f1_micro_test

def evaluate(model, g, inputs, labels, train_nids, val_nids, test_nids, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(g, inputs, batch_size, device)
        # multi gpu
        else:
            pred = model.module.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)

#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    train_mask, val_mask, test_mask, in_feats, labels, n_classes, g = data

    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    test_nid = th.LongTensor(np.nonzero(test_mask)[0])

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')], args.num_negs, args.neg_share)

    # Create PyTorch DataLoader for constructing blocks
    train_seeds = np.arange(g.number_of_edges())
    if n_gpus > 0:
        num_per_gpu = (train_seeds.shape[0] + n_gpus -1) // n_gpus
        train_seeds = train_seeds[proc_id * num_per_gpu :
                                  (proc_id + 1) * num_per_gpu \
                                  if (proc_id + 1) * num_per_gpu < train_seeds.shape[0]
                                  else train_seeds.shape[0]]

    dataloader = DataLoader(
        dataset=train_seeds,
        batch_size=args.batch_size,
        collate_fn=sampler.sample_blocks,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.num_hidden, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.

        tic_step = time.time()
        for step, (pos_graph, neg_graph, blocks) in enumerate(dataloader):
            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            batch_inputs = load_subtensor(g, input_nodes, device)
            d_step = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = time.time()
            pos_edges = pos_graph.number_of_edges()
            neg_edges = neg_graph.number_of_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % args.log_every == 0:
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MiB'.format(
                    proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:]), gpu_mem_alloc))
            tic_step = time.time()

            if step % args.eval_every == 0 and proc_id == 0:
                eval_acc, test_acc = evaluate(model, g, g.ndata['features'], labels, train_nid, val_nid, test_nid, args.batch_size, device)
                print('Eval Acc {:.4f} Test Acc {:.4f}'.format(eval_acc, test_acc))
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                print('Best Eval Acc {:.4f} Test Acc {:.4f}'.format(best_eval_acc, best_test_acc))
        if n_gpus > 1:
            th.distributed.barrier()
    print('Avg epoch time: {}'.format(avg / (epoch - 4)))

def main(args, devices):
    # load reddit data
    data = RedditDataset(self_loop=True)
    train_mask = data.train_mask
    val_mask = data.val_mask
    test_mask = data.test_mask
    features = th.Tensor(data.features)
    in_feats = features.shape[1]
    labels = th.LongTensor(data.labels)
    n_classes = data.num_labels
    # Construct graph
    g = dgl.graph(data.graph.all_edges())
    g.ndata['features'] = features
    # Pack data
    data = train_mask, val_mask, test_mask, in_feats, labels, n_classes, g

    n_gpus = len(devices)
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], data)
    if n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=thread_wrapped_func(run),
                           args=(proc_id, n_gpus, args, devices, data))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()

    run(args, device, data)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument("--gpu", type=str, default='0',
            help="GPU, can be a list of gpus for multi-gpu trianing, e.g., 0,1,2,3; -1 for CPU")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--num-negs', type=int, default=1)
    argparser.add_argument('--neg-share', default=False, action='store_true',
        help="sharing neg nodes for positive nodes")
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=10000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    args = argparser.parse_args()
    
    devices = list(map(int, args.gpu.split(',')))

    main(args, devices)
