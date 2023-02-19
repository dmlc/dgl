import argparse
import collections
import os
import time
import warnings
import zipfile

os.environ["DGLBACKEND"] = "mxnet"
os.environ["MXNET_GPU_MEM_POOL_TYPE"] = "Round"

import dgl
import dgl.data as data
import mxnet as mx
import numpy as np
from mxnet import gluon
from tree_lstm import TreeLSTM

SSTBatch = collections.namedtuple(
    "SSTBatch", ["graph", "mask", "wordid", "label"]
)


def batcher(ctx):
    def batcher_dev(batch):
        batch_trees = dgl.batch(batch)
        return SSTBatch(
            graph=batch_trees,
            mask=batch_trees.ndata["mask"].as_in_context(ctx),
            wordid=batch_trees.ndata["x"].as_in_context(ctx),
            label=batch_trees.ndata["y"].as_in_context(ctx),
        )

    return batcher_dev


def prepare_glove():
    if not (
        os.path.exists("glove.840B.300d.txt")
        and data.utils.check_sha1(
            "glove.840B.300d.txt",
            sha1_hash="294b9f37fa64cce31f9ebb409c266fc379527708",
        )
    ):
        zip_path = data.utils.download(
            "http://nlp.stanford.edu/data/glove.840B.300d.zip",
            sha1_hash="8084fbacc2dee3b1fd1ca4cc534cbfff3519ed0d",
        )
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall()
        if not data.utils.check_sha1(
            "glove.840B.300d.txt",
            sha1_hash="294b9f37fa64cce31f9ebb409c266fc379527708",
        ):
            warnings.warn(
                "The downloaded glove embedding file checksum mismatch. File content "
                "may be corrupted."
            )


def main(args):
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    best_epoch = -1
    best_dev_acc = 0

    cuda = args.gpu >= 0
    if cuda:
        if args.gpu in mx.test_utils.list_gpus():
            ctx = mx.gpu(args.gpu)
        else:
            print(
                "Requested GPU id {} was not found. Defaulting to CPU implementation".format(
                    args.gpu
                )
            )
            ctx = mx.cpu()
    else:
        ctx = mx.cpu()

    if args.use_glove:
        prepare_glove()

    trainset = data.SSTDataset()
    train_loader = gluon.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        batchify_fn=batcher(ctx),
        shuffle=True,
        num_workers=0,
    )
    devset = data.SSTDataset(mode="dev")
    dev_loader = gluon.data.DataLoader(
        dataset=devset,
        batch_size=100,
        batchify_fn=batcher(ctx),
        shuffle=True,
        num_workers=0,
    )

    testset = data.SSTDataset(mode="test")
    test_loader = gluon.data.DataLoader(
        dataset=testset,
        batch_size=100,
        batchify_fn=batcher(ctx),
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
        ctx=ctx,
    )
    print(model)
    params_ex_emb = [
        x
        for x in model.collect_params().values()
        if x.grad_req != "null" and x.shape[0] != trainset.vocab_size
    ]
    params_emb = list(model.embedding.collect_params().values())
    for p in params_emb:
        p.lr_mult = 0.1

    model.initialize(mx.init.Xavier(magnitude=1), ctx=ctx)
    model.hybridize()
    trainer = gluon.Trainer(
        model.collect_params("^(?!embedding).*$"),
        "adagrad",
        {"learning_rate": args.lr, "wd": args.weight_decay},
    )
    trainer_emb = gluon.Trainer(
        model.collect_params("^embedding.*$"),
        "adagrad",
        {"learning_rate": args.lr},
    )

    dur = []
    L = gluon.loss.SoftmaxCrossEntropyLoss(axis=1)
    for epoch in range(args.epochs):
        t_epoch = time.time()
        for step, batch in enumerate(train_loader):
            g = batch.graph
            n = g.number_of_nodes()

            # TODO begin_states function?
            h = mx.nd.zeros((n, args.h_size), ctx=ctx)
            c = mx.nd.zeros((n, args.h_size), ctx=ctx)
            if step >= 3:
                t0 = time.time()  # tik
            with mx.autograd.record():
                pred = model(batch, h, c)
                loss = L(pred, batch.label)

            loss.backward()
            trainer.step(args.batch_size)
            trainer_emb.step(args.batch_size)

            if step >= 3:
                dur.append(time.time() - t0)  # tok

            if step > 0 and step % args.log_every == 0:
                pred = pred.argmax(axis=1).astype(batch.label.dtype)
                acc = (batch.label == pred).sum()
                root_ids = [
                    i
                    for i in range(batch.graph.number_of_nodes())
                    if batch.graph.out_degrees(i) == 0
                ]
                root_acc = np.sum(
                    batch.label.asnumpy()[root_ids] == pred.asnumpy()[root_ids]
                )

                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Acc {:.4f} | Root Acc {:.4f} | Time(s) {:.4f}".format(
                        epoch,
                        step,
                        loss.sum().asscalar(),
                        1.0 * acc.asscalar() / len(batch.label),
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
        for step, batch in enumerate(dev_loader):
            g = batch.graph
            n = g.number_of_nodes()
            h = mx.nd.zeros((n, args.h_size), ctx=ctx)
            c = mx.nd.zeros((n, args.h_size), ctx=ctx)
            pred = model(batch, h, c).argmax(1).astype(batch.label.dtype)

            acc = (batch.label == pred).sum().asscalar()
            accs.append([acc, len(batch.label)])
            root_ids = [
                i
                for i in range(batch.graph.number_of_nodes())
                if batch.graph.out_degrees(i) == 0
            ]
            root_acc = np.sum(
                batch.label.asnumpy()[root_ids] == pred.asnumpy()[root_ids]
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
            model.save_parameters("best_{}.params".format(args.seed))
        else:
            if best_epoch <= epoch - 10:
                break

        # lr decay
        trainer.set_learning_rate(max(1e-5, trainer.learning_rate * 0.99))
        print(trainer.learning_rate)
        trainer_emb.set_learning_rate(
            max(1e-5, trainer_emb.learning_rate * 0.99)
        )
        print(trainer_emb.learning_rate)

    # test
    model.load_parameters("best_{}.params".format(args.seed))
    accs = []
    root_accs = []
    for step, batch in enumerate(test_loader):
        g = batch.graph
        n = g.number_of_nodes()
        h = mx.nd.zeros((n, args.h_size), ctx=ctx)
        c = mx.nd.zeros((n, args.h_size), ctx=ctx)
        pred = model(batch, h, c).argmax(axis=1).astype(batch.label.dtype)

        acc = (batch.label == pred).sum().asscalar()
        accs.append([acc, len(batch.label)])
        root_ids = [
            i
            for i in range(batch.graph.number_of_nodes())
            if batch.graph.out_degrees(i) == 0
        ]
        root_acc = np.sum(
            batch.label.asnumpy()[root_ids] == pred.asnumpy()[root_ids]
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
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--child-sum", action="store_true")
    parser.add_argument("--x-size", type=int, default=300)
    parser.add_argument("--h-size", type=int, default=150)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use-glove", action="store_true")
    args = parser.parse_args()
    print(args)
    main(args)
