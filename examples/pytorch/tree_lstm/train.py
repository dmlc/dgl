import argparse
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import dgl

from tree_lstm import TreeLSTM

def main(args):
    dataset = dgl.data.SST()
    loader = DataLoader(dataset=dataset,
                        batch_size=args.batch_size,
                        collate_fn=dgl.data.SST.batcher,
                        shuffle=False,
                        num_workers=0)
    model = TreeLSTM(dataset.num_vocabs,
                     args.x_size,
                     args.h_size,
                     dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    dur = []
    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            if step >= 3:
                t0 = time.time()
            g = batch.graph
            n = g.number_of_nodes()
            x = th.zeros((n, args.x_size))
            h = th.zeros((n, args.h_size))
            c = th.zeros((n, args.h_size))
            logits = model(batch, x, h, c, train=True)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label) / args.batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 3:
                dur.append(time.time() - t0)

            if step > 0 and step % 5 == 0:
                print("Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
                    epoch, step, loss.item(), np.mean(dur)))
            assert False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=25)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--h-size', type=int, default=512)
    parser.add_argument('--log-every', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--n-ary', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--x-size', type=int, default=256)
    args = parser.parse_args()
    main(args)
