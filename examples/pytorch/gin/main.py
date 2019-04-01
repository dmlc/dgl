import logging
import math
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import GraphDataset
from dataloader import GraphDataLoader, collate
from parser import Parser
from gin import GIN

# loggging basic settings
dtfmt = '[%Y-%m_%d %H:%M:%S]'
fmt = '[LINE:%(lineno)-4d (%(filename)s)][%(levelname)-8s] %(message)s'
logging.basicConfig(level=logging.INFO, format=fmt, datefmt=dtfmt)


def train(args, net, trainloader, optimizer, criterion, epoch):
    net.train()

    total = len(trainloader)

    running_loss = 0
    total_iters = math.ceil(len(trainloader) / args.batch_size)
    pbar = tqdm(range(total_iters), unit='batch')

    logging.debug('the length of trainloader is {}'.format(total))
    logging.debug('the total iters of trainloader is {}'.format(total_iters))

    for pos, (graphs, labels) in zip(pbar, trainloader):
        # batch graphs will be shipped to device in forward part of model
        labels = labels.to(args.device)
        outputs = net(graphs)

        loss = criterion(outputs, labels)
        running_loss += loss.item()

        # backprop
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # report
        pbar.set_description('epoch: %d' % (epoch))

    # the final batch will be aligned
    running_loss = running_loss / total_iters

    return running_loss


def eval_net(args, net, dataloader, type, criterion):
    net.eval()

    total = len(dataloader)
    # tt is better
    tt = 0
    total_loss = 0
    total_correct = 0

    logging.debug('the length of loader is {}'.format(total))

    for data in dataloader:
        graphs, labels = data
        labels = labels.to(args.device)

        tt += len(labels)

        outputs = net(graphs)
        _, predicted = torch.max(outputs.data, 1)

        total_correct += (predicted == labels.data).sum().item()
        loss = criterion(outputs, labels)
        # crossentropy(reduce=True) for default
        total_loss += loss.item() * len(labels)
        logging.debug(
            'loss item is {}; length of labels is {}'
            .format(loss.item(), len(labels)))

    loss, acc = 1.0*total_loss / tt, 1.0*total_correct / tt

    dlname = 'train set' if type == 'train' else 'valid set'
    logging.info(
        '\n{} with learn_eps={} - average loss: {:.4f}, accuracy: {:.0f}%\n'
        .format(dlname, args.learn_eps, loss, 100. * acc))

    net.train()

    return loss, acc


def main(args):

    # set up seeds, args.seed supported
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()

    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=0)
    else:
        args.device = torch.device("cpu")

    dataset = GraphDataset(args.dataset, not args.learn_eps)

    trainloader, validloader = GraphDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        collate_fn=collate, seed=args.seed, shuffle=True,
        split_name='rand', fold_idx=args.fold_idx).train_valid_loader()
    # or split_name='rand', split_ratio=0.7

    model = GIN(
        args.num_layers, args.num_mlp_layers,
        dataset.dim_nfeats, args.hidden_dim, dataset.gclasses,
        args.final_dropout, args.learn_eps,
        args.graph_pooling_type, args.neighbor_pooling_type,
        args.device).to(args.device)

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        train(args, model, trainloader, optimizer, criterion, epoch)

        train_loss, train_acc = eval_net(
            args, model, trainloader, 'train', criterion)
        valid_loss, valid_acc = eval_net(
            args, model, validloader, 'valid', criterion)

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("%f %f %f %f" % (
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc
                ))
                f.write("\n")

        logging.info("the learning eps is: {}".format(model.eps))


if __name__ == '__main__':
    args = Parser(description='GIN').args
    logging.info('show all arguments configuration...')
    logging.info(args)

    main(args)
