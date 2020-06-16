"""Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Reference Code: https://github.com/tkipf/relational-gcn
"""
import argparse
import itertools
import numpy as np
import time
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial

import dgl
from dgl.data.rdf import AIFB, MUTAG, BGS, AM
from model import EntityClassify, RelGraphEmbed

def extract_embed(node_embed, block):
    emb = {}
    for ntype in block.srctypes:
        nid = block.srcnodes[ntype].data[dgl.NID]
        emb[ntype] = node_embed[ntype][nid]
    return emb


def evaluate(model, seeds, blocks, node_embed, labels, category, use_cuda):
    model.eval()
    emb = extract_embed(node_embed, blocks[0])
    lbl = labels[seeds]
    if use_cuda:
        emb = {k : e.cuda() for k, e in emb.items()}
        lbl = lbl.cuda()
    logits = model(emb, blocks)[category]
    loss = F.cross_entropy(logits, lbl)
    acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
    return loss, acc

def main(args):
    # load graph data
    if args.dataset == 'aifb':
        dataset = AIFB()
    elif args.dataset == 'mutag':
        dataset = MUTAG()
    elif args.dataset == 'bgs':
        dataset = BGS()
    elif args.dataset == 'am':
        dataset = AM()
    else:
        raise ValueError()

    g = dataset.graph
    category = dataset.predict_category
    num_classes = dataset.num_classes
    train_idx = dataset.train_idx
    test_idx = dataset.test_idx
    labels = dataset.labels

    # split dataset into train, validate, test
    if args.validation:
        val_idx = train_idx[:len(train_idx) // 5]
        train_idx = train_idx[len(train_idx) // 5:]
    else:
        val_idx = train_idx

    # check cuda
    use_cuda = args.gpu >= 0 and th.cuda.is_available()
    if use_cuda:
        th.cuda.set_device(args.gpu)

    train_label = labels[train_idx]
    val_label = labels[val_idx]
    test_label = labels[test_idx]

    # create embeddings
    embed_layer = RelGraphEmbed(g, args.n_hidden)
    node_embed = embed_layer()
    # create model
    model = EntityClassify(g,
                           args.n_hidden,
                           num_classes,
                           num_bases=args.n_bases,
                           num_hidden_layers=args.n_layers - 2,
                           dropout=args.dropout,
                           use_self_loop=args.use_self_loop)

    if use_cuda:
        model.cuda()

    # train sampler
    sampler = dgl.sampling.MultiLayerNeighborSampler([args.fanout] * args.n_layers)
    collator = dgl.sampling.NodeCollator(g, {category: train_idx}, sampler)
    loader = DataLoader(
        collator.dataset,
        collate_fn=collator.collate,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0)

    # validation sampler
    val_sampler = dgl.sampling.MultiLayerNeighborSampler([None] * args.n_layers)
    val_blocks = val_sampler.sample_blocks(g, {category: val_idx})

    # test sampler
    test_sampler = dgl.sampling.MultiLayerNeighborSampler([None] * args.n_layers)
    test_blocks = test_sampler.sample_blocks(g, {category: test_idx})

    # optimizer
    all_params = itertools.chain(model.parameters(), embed_layer.parameters())
    optimizer = th.optim.Adam(all_params, lr=args.lr, weight_decay=args.l2norm)

    # training loop
    print("start training...")
    dur = []
    for epoch in range(args.n_epochs):
        model.train()
        optimizer.zero_grad()
        if epoch > 3:
            t0 = time.time()

        for i, blocks in enumerate(loader):
            seeds = blocks[-1].dstnodes[category].data[dgl.NID]
            batch_tic = time.time()
            emb = extract_embed(node_embed, blocks[0])
            lbl = labels[seeds]
            if use_cuda:
                emb = {k : e.cuda() for k, e in emb.items()}
                lbl = lbl.cuda()
            logits = model(emb, blocks)[category]
            loss = F.cross_entropy(logits, lbl)
            loss.backward()
            optimizer.step()

            train_acc = th.sum(logits.argmax(dim=1) == lbl).item() / len(seeds)
            print("Epoch {:05d} | Batch {:03d} | Train Acc: {:.4f} | Train Loss: {:.4f} | Time: {:.4f}".
                  format(epoch, i, train_acc, loss.item(), time.time() - batch_tic))

        if epoch > 3:
            dur.append(time.time() - t0)

        val_loss, val_acc = evaluate(model, val_idx, val_blocks, node_embed, labels, category, use_cuda)
        print("Epoch {:05d} | Valid Acc: {:.4f} | Valid loss: {:.4f} | Time: {:.4f}".
              format(epoch, val_acc, val_loss.item(), np.average(dur)))
    print()
    if args.model_path is not None:
        th.save(model.state_dict(), args.model_path)

    test_loss, test_acc = evaluate(model, test_idx, test_blocks, node_embed, labels, category, use_cuda)
    print("Test Acc: {:.4f} | Test loss: {:.4f}".format(test_acc, test_loss.item()))
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=16,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=-1,
            help="number of filter weight matrices, default: -1 [use all]")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("-e", "--n-epochs", type=int, default=20,
            help="number of training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--model_path", type=str, default=None,
            help='path for save the model')
    parser.add_argument("--l2norm", type=float, default=0,
            help="l2 norm coef")
    parser.add_argument("--use-self-loop", default=False, action='store_true',
            help="include self feature as a special relation")
    parser.add_argument("--batch-size", type=int, default=100,
            help="Mini-batch size. If -1, use full graph training.")
    parser.add_argument("--fanout", type=int, default=4,
            help="Fan-out of neighbor sampling.")
    fp = parser.add_mutually_exclusive_group(required=False)
    fp.add_argument('--validation', dest='validation', action='store_true')
    fp.add_argument('--testing', dest='validation', action='store_false')
    parser.set_defaults(validation=True)

    args = parser.parse_args()
    print(args)
    main(args)
