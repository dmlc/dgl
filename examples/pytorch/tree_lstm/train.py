import argparse
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl
import dgl.data as data

from tree_lstm import TreeLSTM

def _batch_to_cuda(batch):
    return data.SSTBatch(graph=batch.graph,
                         nid_with_word = batch.nid_with_word.cuda(),
                         wordid = batch.wordid.cuda(),
                         label = batch.label.cuda())

import dgl.context as ctx
def tensor_topo_traverse(g, cuda, args):
    n = g.number_of_nodes()
    if cuda:
        adjmat = g.cached_graph.adjmat(ctx.gpu(args.gpu))
        mask = th.ones((n, 1)).cuda()
    else:
        adjmat = g.cached_graph.adjmat(ctx.cpu())
        mask = th.ones((n, 1))
    degree = th.spmm(adjmat, mask)
    while th.sum(mask) != 0.:
        v = (degree == 0.).float()
        v = v * mask
        mask = mask - v
        frontier = th.squeeze(th.squeeze(v).nonzero(), 1)
        yield frontier
        degree -= th.spmm(adjmat, v)

def main(args):
    cuda = args.gpu >= 0
    if cuda:
        th.cuda.set_device(args.gpu)
    trainset = data.SST()
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              collate_fn=data.SST.batcher,
                              shuffle=False,
                              num_workers=0)
    #testset = data.SST(mode='test')
    #test_loader = DataLoader(dataset=testset,
    #                         batch_size=100,
    #                         collate_fn=data.SST.batcher,
    #                         shuffle=False,
    #                         num_workers=0)

    model = TreeLSTM(trainset.num_vocabs,
                     args.x_size,
                     args.h_size,
                     trainset.num_classes,
                     args.dropout)
    if cuda:
        model.cuda()
    print(model)
    optimizer = optim.Adagrad(model.parameters(),
                              lr=args.lr,
                              weight_decay=args.weight_decay)
    dur = []
    for epoch in range(args.epochs):
        t_epoch = time.time()
        for step, batch in enumerate(train_loader):
            g = batch.graph
            n = g.number_of_nodes()
            x = th.zeros((n, args.x_size))
            h = th.zeros((n, args.h_size))
            c = th.zeros((n, args.h_size))
            if cuda:
                batch = _batch_to_cuda(batch)
                x = x.cuda()
                h = h.cuda()
                c = c.cuda()

            if step >= 3:
                t0 = time.time()
            # traverse graph
            giter = list(tensor_topo_traverse(g, False, args))
            logits = model(batch, x, h, c, iterator=giter, train=True)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step >= 3:
                dur.append(time.time() - t0)

            if step > 0 and step % args.log_every == 0:
                pred = th.argmax(logits, 1)
                acc = th.sum(th.eq(batch.label, pred))
                mean_dur = np.mean(dur)
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | "
                      "Acc {:.4f} | Time(s) {:.4f} | Trees/s {:.4f}".format(
                    epoch, step, loss.item(), acc.item()/len(batch.label),
                    mean_dur, args.batch_size / mean_dur))
        print("Epoch time(s):", time.time() - t_epoch)

        # test
        #for step, batch in enumerate(test_loader):
        #    g = batch.graph
        #    n = g.number_of_nodes()
        #    x = th.zeros((n, args.x_size))
        #    h = th.zeros((n, args.h_size))
        #    c = th.zeros((n, args.h_size))
        #    logits = model(batch, x, h, c, train=True)
        #    pred = th.argmax(logits, 1)
        #    acc = th.sum(th.eq(batch.label, pred)) / len(batch.label)
        #    print(acc.item())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--x-size', type=int, default=256)
    parser.add_argument('--h-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--n-ary', type=int, default=2)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.5)
    args = parser.parse_args()
    main(args)
