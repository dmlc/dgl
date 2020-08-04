"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/tkipf/relational-gcn
Difference compared to tkipf/relation-gcn
* l2norm applied to all weights
* remove nodes that won't be touched
"""
import argparse
import itertools
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Queue
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
import dgl
from dgl import DGLGraph
from functools import partial

from dgl.data.rdf import AIFBDataset, MUTAGDataset, BGSDataset, AMDataset
from model import RelGraphEmbedLayer
from dgl.nn import RelGraphConv
from utils import thread_wrapped_func

class EntityClassify(nn.Module):
    """ Entity classification class for RGCN
    Parameters
    ----------
    device : int
        Device to run the layer.
    num_nodes : int
        Number of nodes.
    h_dim : int
        Hidden dim size.
    out_dim : int
        Output dim size.
    num_rels : int
        Numer of relation types.
    num_bases : int
        Number of bases. If is none, use number of relations.
    num_hidden_layers : int
        Number of hidden RelGraphConv Layer
    dropout : float
        Dropout
    use_self_loop : bool
        Use self loop if True, default False.
    low_mem : bool
        True to use low memory implementation of relation message passing function
        trade speed with memory consumption
    """
    def __init__(self,
                 device,
                 num_nodes,
                 h_dim,
                 out_dim,
                 num_rels,
                 num_bases=None,
                 num_hidden_layers=1,
                 dropout=0,
                 use_self_loop=False,
                 low_mem=False):
        super(EntityClassify, self).__init__()
        self.device = th.device(device if device >= 0 else 'cpu')
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.low_mem = low_mem

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem))

    def forward(self, blocks, feats, norm=None):
        if blocks is None:
            # full graph training
            blocks = [self.g] * len(self.layers)
        h = feats
        for layer, block in zip(self.layers, blocks):
            block = block.to(self.device)
            h = layer(block, h, block.edata['etype'], block.edata['norm'])
        return h

class NeighborSampler:
    """Neighbor sampler
    Parameters
    ----------
    g : DGLHeterograph
        Full graph
    target_idx : tensor
        The target training node IDs in g
    fanouts : list of int
        Fanout of each hop starting from the seed nodes. If a fanout is None,
        sample full neighbors.
    """
    def __init__(self, g, target_idx, fanouts):
        self.g = g
        self.target_idx = target_idx
        self.fanouts = fanouts

    """Do neighbor sample
    Parameters
    ----------
    seeds :
        Seed nodes
    Returns
    -------
    tensor
        Seed nodes, also known as target nodes
    blocks
        Sampled subgraphs
    """
    def sample_blocks(self, seeds):
        blocks = []
        etypes = []
        norms = []
        ntypes = []
        seeds = th.tensor(seeds).long()
        cur = self.target_idx[seeds]
        for fanout in self.fanouts:
            if fanout is None or fanout == -1:
                frontier = dgl.in_subgraph(self.g, cur)
            else:
                frontier = dgl.sampling.sample_neighbors(self.g, cur, fanout)
            etypes = self.g.edata[dgl.ETYPE][frontier.edata[dgl.EID]]
            norm = self.g.edata['norm'][frontier.edata[dgl.EID]]
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            block.edata['norm'] = norm
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset):
    dev_id = devices[proc_id]
    g, num_of_ntype, num_classes, num_rels, target_idx, \
        train_idx, val_idx, test_idx, labels = dataset

    node_tids = g.ndata[dgl.NTYPE]
    sampler = NeighborSampler(g, target_idx, [args.fanout] * args.n_layers)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=args.num_workers)

    # validation sampler
    val_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    val_loader = DataLoader(dataset=val_idx.numpy(),
                            batch_size=args.batch_size,
                            collate_fn=val_sampler.sample_blocks,
                            shuffle=False,
                            num_workers=args.num_workers)

    # validation sampler
    test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    test_loader = DataLoader(dataset=test_idx.numpy(),
                             batch_size=args.batch_size,
                             collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=args.num_workers)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        backend = 'nccl'
        if args.sparse_embedding:
            backend = 'gloo'
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.
    node_feats = [None] * num_of_ntype
    embed_layer = RelGraphEmbedLayer(dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     sparse_emb=args.sparse_embedding)

    # create model
    model = EntityClassify(dev_id,
                           g.number_of_nodes(),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem)

    if dev_id >= 0:
        th.cuda.set_device(dev_id)
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        # embedding layer may not fit into GPU, then use mix_cpu_gpu
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    # optimizer
    if args.sparse_embedding:
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2norm)
        emb_optimizer = th.optim.SparseAdam(embed_layer.parameters(), lr=args.lr)
    else:
        all_params = itertools.chain(model.parameters(), embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        if args.sparse_embedding:
            emb_optimizer.zero_grad()

        for i, sample_data in enumerate(loader):
            seeds, blocks = sample_data
            t0 = time.time()
            feats = embed_layer(blocks[0].srcdata[dgl.NID].to(dev_id),
                                blocks[0].srcdata[dgl.NTYPE].to(dev_id),
                                node_feats)
            logits = model(blocks, feats)

            loss = F.cross_entropy(logits, labels[seeds])
            t1 = time.time()
            loss.backward()
            optimizer.step()
            if args.sparse_embedding:
                emb_optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:05d}:{:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
                   format(epoch, i, forward_time[-1], backward_time[-1]))
            train_acc = th.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)
            print("Train Accuracy: {:.4f} | Train Loss: {:.4f}".
                  format(train_acc, loss.item()))

        # only process 0 will do the evaluation
        if proc_id == 0:
            model.eval()
            eval_logtis = []
            eval_seeds = []
            for i, sample_data in enumerate(val_loader):
                seeds, blocks = sample_data
                feats = embed_layer(blocks[0].srcdata[dgl.NID].to(dev_id),
                                    blocks[0].srcdata[dgl.NTYPE].to(dev_id),
                                    node_feats)
                logits = model(blocks, feats)
                eval_logtis.append(logits)
                eval_seeds.append(seeds)
            eval_logtis = th.cat(eval_logtis)
            eval_seeds = th.cat(eval_seeds)
            val_loss = F.cross_entropy(eval_logtis, labels[eval_seeds])
            val_acc = th.sum(eval_logtis.argmax(dim=1) == labels[eval_seeds]).item() / len(eval_seeds)
            print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                    format(val_acc, val_loss.item()))
        if n_gpus > 1:
            th.distributed.barrier()
    print()

    # only process 0 will do the testing
    if proc_id == 0:
        model.eval()
        test_logtis = []
        test_seeds = []
        for i, sample_data in enumerate(test_loader):
            seeds, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID].to(dev_id),
                                blocks[0].srcdata[dgl.NTYPE].to(dev_id),
                                [None] * num_of_ntype)
            logits = model(blocks, feats)
            test_logtis.append(logits)
            test_seeds.append(seeds)
        test_logtis = th.cat(test_logtis)
        test_seeds = th.cat(test_seeds)
        test_loss = F.cross_entropy(test_logtis, labels[test_seeds])
        test_acc = th.sum(test_logtis.argmax(dim=1) == labels[test_seeds]).item() / len(test_seeds)
        print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
        print()

    print("{}/{} Mean forward time: {:4f}".format(proc_id, n_gpus,
                                                  np.mean(forward_time[len(forward_time) // 4:])))
    print("{}/{} Mean backward time: {:4f}".format(proc_id, n_gpus,
                                                   np.mean(backward_time[len(backward_time) // 4:])))

def main(args, devices):
    # load graph data
    ogb_dataset = False
    if args.dataset == 'aifb':
        dataset = AIFBDataset()
    elif args.dataset == 'mutag':
        dataset = MUTAGDataset()
    elif args.dataset == 'bgs':
        dataset = BGSDataset()
    elif args.dataset == 'am':
        dataset = AMDataset()
    else:
        raise ValueError()

    # Load from hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    num_of_ntype = len(hg.ntypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_mask = hg.nodes[category].data.pop('train_mask')
    test_mask = hg.nodes[category].data.pop('test_mask')
    labels = hg.nodes[category].data.pop('labels')
    train_idx = th.nonzero(train_mask).squeeze()
    test_idx = th.nonzero(test_mask).squeeze()

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # calculate norm for each edge type and store in edge
    for canonical_etype in hg.canonical_etypes:
        u, v, eid = hg.all_edges(form='all', etype=canonical_etype)
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        hg.edges[canonical_etype].data['norm'] = norm

    # get target category id
    category_id = len(hg.ntypes)
    for i, ntype in enumerate(hg.ntypes):
        if ntype == category:
            category_id = i

    g = dgl.to_homo(hg)
    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()
    node_ids = th.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    target_idx.share_memory_()

    n_gpus = len(devices)
    # cpu
    if devices[0] == -1:
        run(0, 0, args, ['cpu'],
            (g, num_of_ntype, num_classes, num_rels, target_idx,
             train_idx, val_idx, test_idx, labels))
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, args, devices,
            (g, num_of_ntype, num_classes, num_rels, target_idx,
            train_idx, val_idx, test_idx, labels))
    # multi gpu
    else:
        procs = []
        num_train_seeds = train_idx.shape[0]
        tseeds_per_proc = num_train_seeds // n_gpus
        for proc_id in range(n_gpus):
            proc_train_seeds = train_idx[proc_id * tseeds_per_proc :
                                         (proc_id + 1) * tseeds_per_proc \
                                         if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                         else num_train_seeds]
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                                             (g, num_of_ntype, num_classes, num_rels, target_idx,
                                             proc_train_seeds, val_idx, test_idx, labels)))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


def config():
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=str, default='0',
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=50,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--relabel", default=False, action='store_true',
            help="remove untouched nodes and relabel")
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")
    parser.add_argument("--sparse-embedding", action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)
