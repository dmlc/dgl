import argparse
import collections
import time

import dgl

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.nn.init as INIT
import torch.optim as optim
from dgl.data.tree import SSTDataset
from torch.utils.data import DataLoader
from tree_lstm import TreeLSTM

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "mask", "wordid", "label"]
)


def batcher(device):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(
            graph=batch_trees,
            mask=batch_trees.ndata["mask"].to(device),
            wordid=batch_trees.ndata["x"].to(device),
            label=batch_trees.ndata["y"].to(device),
        )

    return batcher_dev


def main(args):
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed(args.seed)

    best_epoch = -1
    best_dev_acc = 0

    cuda = args.gpu >= 0
    device = th.device("cuda:{}".format(args.gpu)) if cuda else th.device("cpu")
    if cuda:
        th.cuda.set_device(args.gpu)

    trainset = SSTDataset()
    train_loader = DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        collate_fn=batcher(device),
        shuffle=True,
        num_workers=0,
    )
    devset = SSTDataset(mode="dev")
    dev_loader = DataLoader(
        dataset=devset,
        batch_size=100,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0,
    )

    testset = SSTDataset(mode="test")
    test_loader = DataLoader(
        dataset=testset,
        batch_size=100,
        collate_fn=batcher(device),
        shuffle=False,
        num_workers=0,
    )

    model = TreeLSTM(
        trainset.vocab_size,
        args.x_size,
        args.h_size,
        trainset.num_classes,
        args.dropout,
        cell_type="childsum" if args.child_sum else "nary",
        pretrained_emb=trainset.pretrained_emb,
    ).to(device)
    print(model)
    params_ex_emb = [
        x
        for x in list(model.parameters())
        if x.requires_grad and x.size(0) != trainset.vocab_size
    ]
    params_emb = list(model.embedding.parameters())

    for p in params_ex_emb:
        if p.dim() > 1:
            INIT.xavier_uniform_(p)

    optimizer = optim.Adagrad(
        [
            {
                "params": params_ex_emb,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
            },
            {"params": params_emb, "lr": 0.1 * args.lr},
        ]
    )

    dur = []
    for epoch in range(args.epochs):
        t_epoch = time.time()
        model.train()
        for step, batch in enumerate(train_loader):
            g = batch.graph.to(device)
            n = g.num_nodes()
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)
            if step >= 3:
                t0 = time.time()  # tik

            logits = model(batch, g, h, c)
            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, batch.label, reduction="sum")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step >= 3:
                dur.append(time.time() - t0)  # tok

            if step > 0 and step % args.log_every == 0:
                pred = th.argmax(logits, 1)
                acc = th.sum(th.eq(batch.label, pred))
                root_ids = [
                    i for i in range(g.num_nodes()) if g.out_degrees(i) == 0
                ]
                root_acc = np.sum(
                    batch.label.cpu().data.numpy()[root_ids]
                    == pred.cpu().data.numpy()[root_ids]
                )

                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} | Root Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch,
                        step,
                        loss.item(),
                        1.0 * acc.item() / len(batch.label),
                        1.0 * root_acc / len(root_ids),
                        np.mean(dur),
                    )
                )
        print(
            "Epoch {:05d} training time {:.4f}s".format(
                epoch, time.time() - t_epoch
            )
        )

        # eval on dev set
        accs = []
        root_accs = []
        model.eval()
        for step, batch in enumerate(dev_loader):
            g = batch.graph.to(device)
            n = g.num_nodes()
            with th.no_grad():
                h = th.zeros((n, args.h_size)).to(device)
                c = th.zeros((n, args.h_size)).to(device)
                logits = model(batch, g, h, c)

            pred = th.argmax(logits, 1)
            acc = th.sum(th.eq(batch.label, pred)).item()
            accs.append([acc, len(batch.label)])
            root_ids = [
                i for i in range(g.num_nodes()) if g.out_degrees(i) == 0
            ]
            root_acc = np.sum(
                batch.label.cpu().data.numpy()[root_ids]
                == pred.cpu().data.numpy()[root_ids]
            )
            root_accs.append([root_acc, len(root_ids)])

        dev_acc = (
            1.0 * np.sum([x[0] for x in accs]) / np.sum([x[1] for x in accs])
        )
        dev_root_acc = (
            1.0
            * np.sum([x[0] for x in root_accs])
            / np.sum([x[1] for x in root_accs])
        )
        print(
            "Epoch {:05d} | Dev Acc {:.4f} | Root Acc {:.4f}".format(
                epoch, dev_acc, dev_root_acc
            )
        )

        if dev_root_acc > best_dev_acc:
            best_dev_acc = dev_root_acc
            best_epoch = epoch
            th.save(model.state_dict(), "best_{}.pkl".format(args.seed))
        else:
            if best_epoch <= epoch - 10:
                break

        # lr decay
        for param_group in optimizer.param_groups:
            param_group["lr"] = max(1e-5, param_group["lr"] * 0.99)  # 10
            print(param_group["lr"])

    # test
    model.load_state_dict(th.load("best_{}.pkl".format(args.seed)))
    accs = []
    root_accs = []
    model.eval()
    for step, batch in enumerate(test_loader):
        g = batch.graph.to(device)
        n = g.num_nodes()
        with th.no_grad():
            h = th.zeros((n, args.h_size)).to(device)
            c = th.zeros((n, args.h_size)).to(device)
            logits = model(batch, g, h, c)

        pred = th.argmax(logits, 1)
        acc = th.sum(th.eq(batch.label, pred)).item()
        accs.append([acc, len(batch.label)])
        root_ids = [i for i in range(g.num_nodes()) if g.out_degrees(i) == 0]
        root_acc = np.sum(
            batch.label.cpu().data.numpy()[root_ids]
            == pred.cpu().data.numpy()[root_ids]
        )
        root_accs.append([root_acc, len(root_ids)])

    test_acc = 1.0 * np.sum([x[0] for x in accs]) / np.sum([x[1] for x in accs])
    test_root_acc = (
        1.0
        * np.sum([x[0] for x in root_accs])
        / np.sum([x[1] for x in root_accs])
    )
    print(
        "------------------------------------------------------------------------------------"
    )
    print(
        "Epoch {:05d} | Test Acc {:.4f} | Root Acc {:.4f}".format(
            best_epoch, test_acc, test_root_acc
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--child-sum", action="store_true")
    parser.add_argument("--x-size", type=int, default=300)
    parser.add_argument("--h-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    args = parser.parse_args()
    print(args)
    main(args)
