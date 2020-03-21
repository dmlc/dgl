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
import tqdm
import traceback
from pyinstrument import Profiler

#### Neighbor sampler

class NeighborSampler(object):
    def __init__(self, g, fanouts):
        self.g = g
        self.fanouts = fanouts

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = dgl.sampling.sample_neighbors(g, seeds, fanout, replace=True)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)
        return blocks

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
            h_dst = h[:block.number_of_nodes('DST/' + block.dsttypes[0])]
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
                h_dst = h[:block.number_of_nodes('DST/' + block.dsttypes[0])]
                h = layer(block, (h, h_dst))
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[start:end] = h.cpu()

            x = y
        return y

def prepare_mp(g):
    """
    Explicitly materialize the CSR, CSC and COO representation of the given graph
    so that they could be shared via copy-on-write to sampler workers and GPU
    trainers.

    This is a workaround before full shared memory support on heterogeneous graphs.
    """
    g.in_degree(0)
    g.out_degree(0)
    g.find_edges([0])

def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

def evaluate(model, g, inputs, labels, val_mask, batch_size, device):
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
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_mask], labels[val_mask])


class Cache:
    def __init__(self, device, cache_size, feats):
        self.device = device
        self.cache_size = cache_size  # cache size in terms of number of nodes
        self.feats = feats
        self.total_size = feats.shape[0]
        self.feat_size = feats.shape[1]

        # a map from node id to cached position
        self.cached_map = th.zeros((self.total_size + 1,)).long() + cache_size
        self.freq = th.zeros((self.total_size + 1,)).long()
        self.cached_nodes = th.zeros((self.cache_size,)).long() + self.total_size
        # each row stores the cached feature
        self.cached_feats = th.zeros((cache_size, self.feat_size)).float().to(device)

        self.count = 0
        self.hit_rate = 0.

    def get(self, nodes):
        if self.cache_size == 0:
            return self.feats[nodes].to(self.device)
        # find hit and miss
        pos = self.cached_map[nodes]
        hit = th.where(pos != self.cache_size)[0]
        miss = th.where(pos == self.cache_size)[0]
        miss_nodes = nodes[miss]

        self.hit_rate = (self.hit_rate * self.count + len(hit) / len(nodes)) / (self.count + 1)
        self.count += 1

        rst = th.zeros((len(nodes), self.feat_size), dtype=self.feats.dtype, device=self.device)
        rst[hit.to(self.device)] = self.cached_feats[pos[hit].to(self.device)]
        miss_feats = self.feats[miss_nodes].to(self.device)
        rst[miss.to(self.device)] = miss_feats

        self.freq[nodes] += 1

        # calculate admit and evict
        miss_and_cached = th.cat([miss_nodes, self.cached_nodes])
        all_freq = self.freq[miss_and_cached]
        _, sorted_pos = th.sort(all_freq, descending=True)
        keep = sorted_pos[:self.cache_size]
        evict = sorted_pos[self.cache_size:]
        admit = keep[keep < len(miss_nodes)]
        evict = evict[evict >= len(miss_nodes)] - len(miss_nodes)
        #print(evict)

        # update cache
        admit_nodes = miss_nodes[admit]
        #print('admit nodes', th.sort(admit_nodes)[0])
        #print('evict pos', evict[th.sort(admit_nodes)[1]])
        evict_nodes = self.cached_nodes[evict]
        self.cached_map[admit_nodes] = evict
        self.cached_map[evict_nodes] = self.cache_size
        self.cached_nodes[evict] = admit_nodes
        self.cached_feats[evict] = miss_feats[admit]

        #print(self.cached_feats[self.cached_map[nodes]])
        #print('admit', len(admit), admit_nodes)
        return rst

def load_subtensor(g, labels, seeds, input_nodes, device, cache):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    #batch_inputs = g.ndata['features'][input_nodes].to(device)
    batch_inputs = cache.get(input_nodes)
    batch_labels = labels[seeds].to(device)
    return batch_inputs, batch_labels

#### Entry point
def run(args, device, data):
    # Unpack data
    train_mask, val_mask, in_feats, labels, n_classes, g = data
    train_nid = th.LongTensor(np.nonzero(train_mask)[0])
    val_nid = th.LongTensor(np.nonzero(val_mask)[0])
    train_mask = th.BoolTensor(train_mask)
    val_mask = th.BoolTensor(val_mask)

    # Create sampler
    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')])

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

    # Feature cache
    cache = Cache(device, args.cache_size, g.ndata['features'])

    # Training loop
    avg = 0
    iter_tput = []
    #profiler = Profiler()
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        extract_time = 0.
        extract_GB = 0.
        train_time = 0.
        copy_time = 0.
        for step, blocks in enumerate(dataloader):
            tic_step = time.time()

            # The nodes for input lies at the LHS side of the first block.
            # The nodes for output lies at the RHS side of the last block.
            input_nodes = blocks[0].srcdata[dgl.NID]
            seeds = blocks[-1].dstdata[dgl.NID]

            # Load the input features as well as output labels
            t0 = time.time()
            batch_inputs, batch_labels = load_subtensor(g, labels, seeds, input_nodes, device, cache)
            extract_time += time.time() - t0
            extract_GB += batch_inputs.numel() * 4 / 1000 / 1000 / 1000

            # Compute loss and prediction
            t0 = time.time()
            #profiler.start()
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss.item()
            #profiler.stop()
            train_time += time.time() - t0

            iter_tput.append(len(seeds) / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | Speed (samples/sec) {:.4f} | GPU {:.1f} MiB'.format(
                    epoch, step, loss.item(), acc.item(), np.mean(iter_tput[3:]), gpu_mem_alloc))

        toc = time.time()
        print('Epoch Time(s): {:.4f}'.format(toc - tic))
        print('Extract Time(s): {:.4f} | Copy Time(s): {:.4f} | Amount(GB): {:.4f} | Train Time (s): {:.4f}'.format(extract_time, copy_time, extract_GB, train_time))
        print('Avg cache hit rate: {:.4f}'.format(cache.hit_rate))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc = evaluate(model, g, g.ndata['features'], labels, val_mask, args.batch_size, device)
            print('Eval Acc {:.4f}'.format(eval_acc))
    #print(profiler.output_text(unicode=True, color=True))

    print('Avg epoch time: {:.4f}'.format(avg / (epoch - 4)))

if __name__ == '__main__':
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=0,
        help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--num-epochs', type=int, default=20)
    argparser.add_argument('--num-hidden', type=int, default=16)
    argparser.add_argument('--num-layers', type=int, default=2)
    argparser.add_argument('--fan-out', type=str, default='10,25')
    argparser.add_argument('--batch-size', type=int, default=1000)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=5)
    argparser.add_argument('--lr', type=float, default=0.003)
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument('--num-workers', type=int, default=0,
        help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--cache-size', type=int, default=10000,
        help="Node feature GPU cache size (number of nodes)")
    args = argparser.parse_args()
    
    if args.gpu >= 0:
        device = th.device('cuda:%d' % args.gpu)
    else:
        device = th.device('cpu')

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
    prepare_mp(g)
    # Pack data
    data = train_mask, val_mask, in_feats, labels, n_classes, g

    run(args, device, data)
