import sys
from parser import Parser

import mxnet as mx
import numpy as np
from dataloader import collate, GraphDataLoader

from dgl.data.gindt import GINDataset
from gin import GIN
from mxnet import gluon, nd
from mxnet.gluon import nn
from tqdm import tqdm


def train(args, net, trainloader, trainer, criterion, epoch):
    running_loss = 0
    total_iters = len(trainloader)
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(total_iters), unit="batch", position=2, file=sys.stdout)

    for pos, (graphs, labels) in zip(bar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.as_in_context(args.device)
        feat = graphs.ndata["attr"].as_in_context(args.device)

        with mx.autograd.record():
            graphs = graphs.to(args.device)
            outputs = net(graphs, feat)
            loss = criterion(outputs, labels)
            loss = loss.sum() / len(labels)

        running_loss += loss.asscalar()

        # backprop
        loss.backward()
        trainer.step(batch_size=1)

        # report
        bar.set_description("epoch-{}".format(epoch))
    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, criterion):
    total = 0
    total_loss = 0
    total_correct = 0

    for data in dataloader:
        graphs, labels = data
        labels = labels.as_in_context(args.device)
        feat = graphs.ndata["attr"].as_in_context(args.device)

        total += len(labels)
        graphs = graphs.to(args.device)
        outputs = net(graphs, feat)
        predicted = nd.argmax(outputs, axis=1)
        predicted = predicted.astype("int64")

        total_correct += (predicted == labels).sum().asscalar()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.sum().asscalar()

    loss, acc = 1.0 * total_loss / total, 1.0 * total_correct / total

    return loss, acc


def main(args):
    # set up seeds, args.seed supported
    mx.random.seed(0)
    np.random.seed(seed=0)

    if args.device >= 0:
        args.device = mx.gpu(args.device)
    else:
        args.device = mx.cpu()

    dataset = GINDataset(args.dataset, not args.learn_eps)

    trainloader, validloader = GraphDataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate,
        seed=args.seed,
        shuffle=True,
        split_name="fold10",
        fold_idx=args.fold_idx,
    ).train_valid_loader()
    # or split_name='rand', split_ratio=0.7

    model = GIN(
        args.num_layers,
        args.num_mlp_layers,
        dataset.dim_nfeats,
        args.hidden_dim,
        dataset.gclasses,
        args.final_dropout,
        args.learn_eps,
        args.graph_pooling_type,
        args.neighbor_pooling_type,
    )
    model.initialize(ctx=args.device)

    criterion = gluon.loss.SoftmaxCELoss()

    print(model.collect_params())
    lr_scheduler = mx.lr_scheduler.FactorScheduler(50, 0.5)
    trainer = gluon.Trainer(
        model.collect_params(), "adam", {"lr_scheduler": lr_scheduler}
    )

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(
        range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout
    )
    vbar = tqdm(
        range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout
    )
    lrbar = tqdm(
        range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout
    )

    for epoch, _, _ in zip(tbar, vbar, lrbar):
        train(args, model, trainloader, trainer, criterion, epoch)

        train_loss, train_acc = eval_net(args, model, trainloader, criterion)
        tbar.set_description(
            "train set - average loss: {:.4f}, accuracy: {:.0f}%".format(
                train_loss, 100.0 * train_acc
            )
        )

        valid_loss, valid_acc = eval_net(args, model, validloader, criterion)
        vbar.set_description(
            "valid set - average loss: {:.4f}, accuracy: {:.0f}%".format(
                valid_loss, 100.0 * valid_acc
            )
        )

        if not args.filename == "":
            with open(args.filename, "a") as f:
                f.write(
                    "%s %s %s %s"
                    % (
                        args.dataset,
                        args.learn_eps,
                        args.neighbor_pooling_type,
                        args.graph_pooling_type,
                    )
                )
                f.write("\n")
                f.write(
                    "%f %f %f %f"
                    % (train_loss, train_acc, valid_loss, valid_acc)
                )
                f.write("\n")

        lrbar.set_description(
            "Learning eps with learn_eps={}: {}".format(
                args.learn_eps,
                [
                    layer.eps.data(args.device).asscalar()
                    for layer in model.ginlayers
                ],
            )
        )

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == "__main__":
    args = Parser(description="GIN").args
    print("show all arguments configuration...")
    print(args)

    main(args)
