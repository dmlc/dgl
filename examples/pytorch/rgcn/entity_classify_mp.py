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
import gc
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
import tqdm 

from ogb.nodeproppred import DglNodePropPredDataset

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
                 low_mem=False,
                 layer_norm=False):
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
        self.layer_norm = layer_norm

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(RelGraphConv(
            self.h_dim, self.h_dim, self.num_rels, "basis",
            self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
            low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2h
        for idx in range(self.num_hidden_layers):
            self.layers.append(RelGraphConv(
                self.h_dim, self.h_dim, self.num_rels, "basis",
                self.num_bases, activation=F.relu, self_loop=self.use_self_loop,
                low_mem=self.low_mem, dropout=self.dropout, layer_norm = layer_norm))
        # h2o
        self.layers.append(RelGraphConv(
            self.h_dim, self.out_dim, self.num_rels, "basis",
            self.num_bases, activation=None,
            self_loop=self.use_self_loop,
            low_mem=self.low_mem, layer_norm = layer_norm))

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
            block = dgl.to_block(frontier, cur)
            block.srcdata[dgl.NTYPE] = self.g.ndata[dgl.NTYPE][block.srcdata[dgl.NID]]
            block.srcdata['type_id'] = self.g.ndata[dgl.NID][block.srcdata[dgl.NID]]
            block.edata['etype'] = etypes
            cur = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seeds, blocks

def evaluate(model, embed_layer, eval_loader, node_feats):
    model.eval()
    embed_layer.eval()
    eval_logits = []
    eval_seeds = []
 
    with th.no_grad():
        for sample_data in tqdm.tqdm(eval_loader):
            th.cuda.empty_cache()
            seeds, blocks = sample_data
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                    blocks[0].srcdata[dgl.NTYPE],
                    blocks[0].srcdata['type_id'],
                    node_feats)
            logits = model(blocks, feats)
            eval_logits.append(logits.cpu().detach())
            eval_seeds.append(seeds.cpu().detach())
    eval_logits = th.cat(eval_logits)
    eval_seeds = th.cat(eval_seeds)
 
    return eval_logits, eval_seeds


@thread_wrapped_func
def run(proc_id, n_gpus, args, devices, dataset, split, queue=None):
    dev_id = devices[proc_id]
    g, node_feats, num_of_ntype, num_classes, num_rels, target_idx, \
        train_idx, val_idx, test_idx, labels = dataset
    if split is not None:
        train_seed, val_seed, test_seed = split
        train_idx = train_idx[train_seed]
        val_idx = val_idx[val_seed]
        test_idx = test_idx[test_seed]

    fanouts = [int(fanout) for fanout in args.fanout.split(',')]
    node_tids = g.ndata[dgl.NTYPE]
    sampler = NeighborSampler(g, target_idx, fanouts)
    loader = DataLoader(dataset=train_idx.numpy(),
                        batch_size=args.batch_size,
                        collate_fn=sampler.sample_blocks,
                        shuffle=True,
                        num_workers=args.num_workers)

    # validation sampler
    val_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    val_loader = DataLoader(dataset=val_idx.numpy(),
                            batch_size=args.eval_batch_size,
                            collate_fn=val_sampler.sample_blocks,
                            shuffle=False,
                            num_workers=args.num_workers)

    # validation sampler
    test_sampler = NeighborSampler(g, target_idx, [None] * args.n_layers)
    test_loader = DataLoader(dataset=test_idx.numpy(),
                             batch_size=args.eval_batch_size,
                             collate_fn=test_sampler.sample_blocks,
                             shuffle=False,
                             num_workers=args.num_workers)

    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        backend = 'nccl'

        # using sparse embedding or usig mix_cpu_gpu model (embedding model can not be stored in GPU)
        if args.sparse_embedding or args.mix_cpu_gpu:
            backend = 'gloo'
        th.distributed.init_process_group(backend=backend,
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=dev_id)

    # node features
    # None for one-hot feature, if not none, it should be the feature tensor.
    # 
    embed_layer = RelGraphEmbedLayer(dev_id,
                                     g.number_of_nodes(),
                                     node_tids,
                                     num_of_ntype,
                                     node_feats,
                                     args.n_hidden,
                                     sparse_emb=args.sparse_embedding)

    # create model
    # all model params are in device.
    model = EntityClassify(dev_id,
                           g.number_of_nodes(),
                           args.n_hidden,
                           num_classes,
                           num_rels,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop,
                           low_mem=args.low_mem,
                           layer_norm=args.layer_norm)

    if dev_id >= 0 and n_gpus == 1:
        th.cuda.set_device(dev_id)
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        # embedding layer may not fit into GPU, then use mix_cpu_gpu
        if args.mix_cpu_gpu is False:
            embed_layer.cuda(dev_id)

    if n_gpus > 1:
        labels = labels.to(dev_id)
        model.cuda(dev_id)
        if args.mix_cpu_gpu:
            embed_layer = DistributedDataParallel(embed_layer, device_ids=None, output_device=None)
        else:
            embed_layer.cuda(dev_id)
            embed_layer = DistributedDataParallel(embed_layer, device_ids=[dev_id], output_device=dev_id)
        model = DistributedDataParallel(model, device_ids=[dev_id], output_device=dev_id)

    # optimizer
    if args.sparse_embedding:
        dense_params = list(model.parameters())
        if args.node_feats:
            if  n_gpus > 1:
                dense_params += list(embed_layer.module.embeds.parameters())
            else:
                dense_params += list(embed_layer.embeds.parameters())
        optimizer = th.optim.Adam(dense_params, lr=args.lr, weight_decay=args.l2norm)
        if  n_gpus > 1:
            emb_optimizer = th.optim.SparseAdam(embed_layer.module.node_embeds.parameters(), lr=args.lr)
        else:
            emb_optimizer = th.optim.SparseAdam(embed_layer.node_embeds.parameters(), lr=args.lr)
    else:
        all_params = list(model.parameters()) + list(embed_layer.parameters())
        optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    forward_time = []
    backward_time = []

    for epoch in range(args.n_epochs):
        model.train()
        embed_layer.train()

        for i, sample_data in enumerate(loader):
            seeds, blocks = sample_data
            t0 = time.time()
            feats = embed_layer(blocks[0].srcdata[dgl.NID],
                                blocks[0].srcdata[dgl.NTYPE],
                                blocks[0].srcdata['type_id'],
                                node_feats)
            logits = model(blocks, feats)
            loss = F.cross_entropy(logits, labels[seeds])
            t1 = time.time()
            optimizer.zero_grad()
            if args.sparse_embedding:
                emb_optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            if args.sparse_embedding:
                emb_optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            train_acc = th.sum(logits.argmax(dim=1) == labels[seeds]).item() / len(seeds)
            if i % 100 and proc_id == 0:
                print("Train Accuracy: {:.4f} | Train Loss: {:.4f}".
                    format(train_acc, loss.item()))
        print("Epoch {:05d}:{:05d} | Train Forward Time(s) {:.4f} | Backward Time(s) {:.4f}".
            format(epoch, i, forward_time[-1], backward_time[-1]))

        if (queue is not None) or (proc_id == 0):
            val_logits, val_seeds = evaluate(model, embed_layer, val_loader, node_feats)
            if queue is not None:
                queue.put((val_logits, val_seeds))

            # gather evaluation result from multiple processes
            if proc_id == 0:
                if queue is not None:
                    val_logits = []
                    val_seeds = []
                    for i in range(n_gpus):
                        log = queue.get()
                        val_l, val_s = log
                        val_logits.append(val_l)
                        val_seeds.append(val_s)
                    val_logits = th.cat(val_logits)
                    val_seeds = th.cat(val_seeds)
                val_loss = F.cross_entropy(val_logits, labels[val_seeds].cpu()).item()
                val_acc = th.sum(val_logits.argmax(dim=1) == labels[val_seeds].cpu()).item() / len(val_seeds)

                print("Validation Accuracy: {:.4f} | Validation loss: {:.4f}".
                        format(val_acc, val_loss))
        if n_gpus > 1:
            th.distributed.barrier()

    # only process 0 will do the evaluation
    if (queue is not None) or (proc_id == 0):
        test_logits, test_seeds = evaluate(model, embed_layer, test_loader, node_feats)
        if queue is not None:
            queue.put((test_logits, test_seeds))

        # gather evaluation result from multiple processes
        if proc_id == 0:
            if queue is not None:
                test_logits = []
                test_seeds = []
                for i in range(n_gpus):
                    log = queue.get()
                    test_l, test_s = log
                    test_logits.append(test_l)
                    test_seeds.append(test_s)
                test_logits = th.cat(test_logits)
                test_seeds = th.cat(test_seeds)
            test_loss = F.cross_entropy(test_logits, labels[test_seeds].cpu()).item()
            test_acc = th.sum(test_logits.argmax(dim=1) == labels[test_seeds].cpu()).item() / len(test_seeds)
            print("Test Accuracy: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss))
            print()

    # sync for test
    if n_gpus > 1:
        th.distributed.barrier()

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
    elif args.dataset == 'ogbn-mag':
        dataset = DglNodePropPredDataset(name=args.dataset)
        ogb_dataset = True
    else:
        raise ValueError()

    if ogb_dataset is True:
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]['paper']
        val_idx = split_idx["valid"]['paper']
        test_idx = split_idx["test"]['paper']
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], 'rev-'+etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes['paper'].data['feat'] = hg_orig.nodes['paper'].data['feat']
        labels = labels['paper'].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        if args.dataset == 'ogbn-mag':
            category = 'paper'
        print('Number of relations: {}'.format(num_rels))
        print('Number of class: {}'.format(num_classes))
        print('Number of train: {}'.format(len(train_idx)))
        print('Number of valid: {}'.format(len(val_idx)))
        print('Number of test: {}'.format(len(test_idx)))

        if args.node_feats:
            node_feats = []
            for ntype in hg.ntypes:
                if len(hg.nodes[ntype].data) == 0:
                    node_feats.append(None)
                else:
                    assert len(hg.nodes[ntype].data) == 1
                    feat = hg.nodes[ntype].data.pop('feat')
                    node_feats.append(feat.share_memory_())
        else:
            node_feats = [None] * num_of_ntype
    else:
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
        node_feats = [None] * num_of_ntype

        # AIFB, MUTAG, BGS and AM datasets do not provide validation set split.
        # Split train set into train and validation if args.validation is set
        # otherwise use train set as the validation set.
        if args.validation:
            val_idx = train_idx[:len(train_idx) // 5]
            train_idx = train_idx[len(train_idx) // 5:]
        else:
            val_idx = train_idx

    # calculate norm for each edge type and store in edge
    if args.global_norm is False:
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

    g = dgl.to_homogeneous(hg, edata=['norm'])
    if args.global_norm:
        u, v, eid = g.all_edges(form='all')
        _, inverse_index, count = th.unique(v, return_inverse=True, return_counts=True)
        degrees = count[inverse_index]
        norm = th.ones(eid.shape[0]) / degrees
        norm = norm.unsqueeze(1)
        g.edata['norm'] = norm

    g.ndata[dgl.NTYPE].share_memory_()
    g.edata[dgl.ETYPE].share_memory_()
    g.edata['norm'].share_memory_()
    node_ids = th.arange(g.number_of_nodes())

    # find out the target node ids
    node_tids = g.ndata[dgl.NTYPE]
    loc = (node_tids == category_id)
    target_idx = node_ids[loc]
    target_idx.share_memory_()
    train_idx.share_memory_()
    val_idx.share_memory_()
    test_idx.share_memory_()

    n_gpus = len(devices)
    # cpu
    if devices[0] == -1:
        run(0, 0, args, ['cpu'],
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
             train_idx, val_idx, test_idx, labels), None, None)
    # gpu
    elif n_gpus == 1:
        run(0, n_gpus, args, devices,
            (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
            train_idx, val_idx, test_idx, labels), None, None)
    # multi gpu
    else:
        queue = mp.Queue(n_gpus)
        procs = []
        num_train_seeds = train_idx.shape[0]
        num_valid_seeds = val_idx.shape[0]
        num_test_seeds = test_idx.shape[0]
        train_seeds = th.randperm(num_train_seeds)
        valid_seeds = th.randperm(num_valid_seeds)
        test_seeds = th.randperm(num_test_seeds)
        tseeds_per_proc = num_train_seeds // n_gpus
        vseeds_per_proc = num_valid_seeds // n_gpus
        tstseeds_per_proc = num_test_seeds // n_gpus
        for proc_id in range(n_gpus):
            # we have multi-gpu for training, evaluation and testing
            # so split trian set, valid set and test set into num-of-gpu parts.
            proc_train_seeds = train_seeds[proc_id * tseeds_per_proc :
                                           (proc_id + 1) * tseeds_per_proc \
                                           if (proc_id + 1) * tseeds_per_proc < num_train_seeds \
                                           else num_train_seeds]
            proc_valid_seeds = valid_seeds[proc_id * vseeds_per_proc :
                                           (proc_id + 1) * vseeds_per_proc \
                                           if (proc_id + 1) * vseeds_per_proc < num_valid_seeds \
                                           else num_valid_seeds]
            proc_test_seeds = test_seeds[proc_id * tstseeds_per_proc :
                                         (proc_id + 1) * tstseeds_per_proc \
                                         if (proc_id + 1) * tstseeds_per_proc < num_test_seeds \
                                         else num_test_seeds]
            p = mp.Process(target=run, args=(proc_id, n_gpus, args, devices,
                                             (g, node_feats, num_of_ntype, num_classes, num_rels, target_idx,
                                             train_idx, val_idx, test_idx, labels),
                                             (proc_train_seeds, proc_valid_seeds, proc_test_seeds),
                                             queue))
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
    parser.add_argument("--fanout", type=str, default="4, 4",
            help="Fan-out of neighbor sampling.")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. ")
    parser.add_argument("--eval-batch-size", type=int, default=128,
            help="Mini-batch size. ")
    parser.add_argument("--num-workers", type=int, default=0,
            help="Number of workers for dataloader.")
    parser.add_argument("--low-mem", default=False, action='store_true',
            help="Whether use low mem RelGraphCov")
    parser.add_argument("--mix-cpu-gpu", default=False, action='store_true',
            help="Whether store node embeddins in cpu")
    parser.add_argument("--sparse-embedding", action='store_true',
            help='Use sparse embedding for node embeddings.')
    parser.add_argument('--node-feats', default=False, action='store_true',
            help='Whether use node features')
    parser.add_argument('--global-norm', default=False, action='store_true',
            help='User global norm instead of per node type norm')
    parser.add_argument('--layer-norm', default=False, action='store_true',
            help='Use layer norm')
    parser.set_defaults(validation=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = config()
    devices = list(map(int, args.gpu.split(',')))
    print(args)
    main(args, devices)
