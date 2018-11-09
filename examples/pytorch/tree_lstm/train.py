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

rand_seed = 12110 
np.random.seed(rand_seed)
th.manual_seed(rand_seed)
th.cuda.manual_seed(rand_seed)

def _batch_to_cuda(batch):
    return data.SSTBatch(graph=batch.graph,
                         nid_with_word = batch.nid_with_word.cuda(),
                         wordid = batch.wordid.cuda(),
                         label = batch.label.cuda())

def main(args):
    cuda = args.gpu >= 0
    if cuda:
        th.cuda.set_device(args.gpu)

    trainset = data.SST()
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              collate_fn=data.SST.batcher,
                              shuffle=True,
                              num_workers=0)
    devset = data.SST(mode='dev')
    dev_loader = DataLoader(dataset=devset,
                             batch_size=100,
                             collate_fn=data.SST.batcher,
                             shuffle=False,
                             num_workers=0)

    testset = data.SST(mode='test')
    test_loader = DataLoader(dataset=testset,
                             batch_size=100,
                             collate_fn=data.SST.batcher,
                             shuffle=False,
                             num_workers=0)

    model = TreeLSTM(trainset.num_vocabs,
                     args.x_size,
                     args.h_size,
                     trainset.num_classes,
                     args.dropout,
                     pretrained_emb = trainset.pretrained_emb)
    if cuda:
        model.cuda()
    print(model)
    params_ex_emb =[x for x in list(model.parameters()) if x.requires_grad and x.size(0)!=trainset.num_vocabs]
    params_emb = list(model.embedding.parameters())

    optimizer = optim.Adagrad([{'params':params_ex_emb, 'lr':args.lr, 'weight_decay':args.weight_decay}, {'params':params_emb, 'lr':args.lr*0.1}])
    dur = []
    for epoch in range(args.epochs):
        t_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            if cuda:
                batch = _batch_to_cuda(batch)
            g = batch.graph
            n = g.number_of_nodes()
            x = th.zeros((n, args.h_size * 3))
            h = th.zeros((n, args.h_size))
            c = th.zeros((n, args.h_size))
            if cuda:
                x = x.cuda()
                h = h.cuda()
                c = c.cuda()

            if step >= 3:
                t0 = time.time()

            # traverse graph
            #giter = list(tensor_topo_traverse(g, False, args))
            logits = model(batch, x, h, c, iterator=False, train=True)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, size_average=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step >= 3:
                dur.append(time.time() - t0)

            if step > 0 and step % args.log_every == 0:
                pred = th.argmax(logits, 1)
                acc = th.sum(th.eq(batch.label, pred))
                #root_ids = [x for x in batch.graph if batch.graph.out_degree(x)==0]
                #root_acc = np.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} | Root Acc {:.4f} | Time(s) {:.4f}".format(
                    epoch, step, loss.item(), 1.0*acc.item()/len(batch.label), 0, np.mean(dur)))# 1.0*root_acc/len(root_ids), np.mean(dur)))

        # test on dev set
        accs = []
        root_accs = []
        model.eval()
        for step, batch in enumerate(dev_loader):
            if cuda:
                batch = _batch_to_cuda(batch)
            g = batch.graph
            n = g.number_of_nodes()
            x = th.zeros((n, args.h_size * 3))
            h = th.zeros((n, args.h_size))
            c = th.zeros((n, args.h_size))
            if cuda:
                x = x.cuda()
                h = h.cuda()
                c = c.cuda()

            # traverse graph
            logits = model(batch, x, h, c, iterator=False, train=True)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])
            root_ids = [x for x in batch.graph if batch.graph.out_degree(x)==0]
            root_acc = np.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
            root_accs.append([root_acc, len(root_ids)])
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10

        print("Epoch {:05d} | Dev Acc {:.4f} | Root Acc {:.4f}".format(epoch, 1.0*np.sum([x[0] for x in accs])/np.sum([x[1] for x in accs]), 1.0*np.sum([x[0] for x in root_accs])/np.sum([x[1] for x in root_accs])  ))

        # test
        accs = []
        root_accs = []
        model.eval()
        for step, batch in enumerate(test_loader):
            if cuda:
                batch = _batch_to_cuda(batch)
            g = batch.graph
            n = g.number_of_nodes()
            x = th.zeros((n, args.h_size * 3))
            h = th.zeros((n, args.h_size))
            c = th.zeros((n, args.h_size))
            if cuda:
                x = x.cuda()
                h = h.cuda()
                c = c.cuda()

            # traverse graph
            logits = model(batch, x, h, c, iterator=None, train=True)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])
            root_ids = [x for x in batch.graph if batch.graph.out_degree(x)==0]
            root_acc = np.sum(batch.label.cpu().data.numpy()[root_ids] == pred.cpu().data.numpy()[root_ids])
            root_accs.append([root_acc, len(root_ids)])
        #lr decay
        for param_group in optimizer.param_groups:
            param_group['lr'] = max(1e-5, param_group['lr']*0.99) #10

        print("Epoch {:05d} | Test Acc {:.4f} | Root Acc {:.4f}".format(epoch, 1.0*np.sum([x[0] for x in accs])/np.sum([x[1] for x in accs]), 1.0*np.sum([x[0] for x in root_accs])/np.sum([x[1] for x in root_accs])  ))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--x-size', type=int, default=300)
    parser.add_argument('--h-size', type=int, default=150)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--log-every', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--dropout', type=float, default=0.3)
    args = parser.parse_args()
    print(args)
    main(args)
