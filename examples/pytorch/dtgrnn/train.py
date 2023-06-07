import argparse
from functools import partial

import dgl

import numpy as np
import torch
import torch.nn as nn
from dataloading import (
    METR_LAGraphDataset,
    METR_LATestDataset,
    METR_LATrainDataset,
    METR_LAValidDataset,
    PEMS_BAYGraphDataset,
    PEMS_BAYTestDataset,
    PEMS_BAYTrainDataset,
    PEMS_BAYValidDataset,
)
from dcrnn import DiffConv
from gaan import GatedGAT
from model import GraphRNN
from torch.utils.data import DataLoader
from utils import get_learning_rate, masked_mae_loss, NormalizationLayer

batch_cnt = [0]


def train(
    model,
    graph,
    dataloader,
    optimizer,
    scheduler,
    normalizer,
    loss_fn,
    device,
    args,
):
    total_loss = []
    graph = graph.to(device)
    model.train()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        optimizer.zero_grad()
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            y_buff[: x.shape[0], :, :, :] = y
            y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff
            y = y_buff
        # Permute the dimension for shaping
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        x_norm = (
            normalizer.normalize(x)
            .reshape(x.shape[0], -1, x.shape[3])
            .float()
            .to(device)
        )
        y_norm = (
            normalizer.normalize(y)
            .reshape(x.shape[0], -1, x.shape[3])
            .float()
            .to(device)
        )
        y = y.reshape(y.shape[0], -1, y.shape[3]).float().to(device)

        batch_graph = dgl.batch([graph] * batch_size)
        output = model(batch_graph, x_norm, y_norm, batch_cnt[0], device)
        # Denormalization for loss compute
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if get_learning_rate(optimizer) > args.minimum_lr:
            scheduler.step()
        total_loss.append(float(loss))
        batch_cnt[0] += 1
        print("\rBatch: ", i, end="")
    return np.mean(total_loss)


def eval(model, graph, dataloader, normalizer, loss_fn, device, args):
    total_loss = []
    graph = graph.to(device)
    model.eval()
    batch_size = args.batch_size
    for i, (x, y) in enumerate(dataloader):
        # Padding: Since the diffusion graph is precmputed we need to pad the batch so that
        # each batch have same batch size
        if x.shape[0] != batch_size:
            x_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            y_buff = torch.zeros(batch_size, x.shape[1], x.shape[2], x.shape[3])
            x_buff[: x.shape[0], :, :, :] = x
            x_buff[x.shape[0] :, :, :, :] = x[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            y_buff[: x.shape[0], :, :, :] = y
            y_buff[x.shape[0] :, :, :, :] = y[-1].repeat(
                batch_size - x.shape[0], 1, 1, 1
            )
            x = x_buff
            y = y_buff
        # Permute the order of dimension
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)

        x_norm = (
            normalizer.normalize(x)
            .reshape(x.shape[0], -1, x.shape[3])
            .float()
            .to(device)
        )
        y_norm = (
            normalizer.normalize(y)
            .reshape(x.shape[0], -1, x.shape[3])
            .float()
            .to(device)
        )
        y = y.reshape(x.shape[0], -1, x.shape[3]).to(device)

        batch_graph = dgl.batch([graph] * batch_size)
        output = model(batch_graph, x_norm, y_norm, i, device)
        y_pred = normalizer.denormalize(output)
        loss = loss_fn(y_pred, y)
        total_loss.append(float(loss))
    return np.mean(total_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Define the arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Size of batch for minibatch Training",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for parallel dataloading",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="dcrnn",
        help="WHich model to use DCRNN vs GaAN",
    )
    parser.add_argument(
        "--gpu", type=int, default=-1, help="GPU indexm -1 for CPU training"
    )
    parser.add_argument(
        "--diffsteps",
        type=int,
        default=2,
        help="Step of constructing the diffusiob matrix",
    )
    parser.add_argument(
        "--num_heads", type=int, default=2, help="Number of multiattention head"
    )
    parser.add_argument(
        "--decay_steps",
        type=int,
        default=2000,
        help="Teacher forcing probability decay ratio",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, help="Initial learning rate"
    )
    parser.add_argument(
        "--minimum_lr",
        type=float,
        default=2e-6,
        help="Lower bound of learning rate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="LA",
        help="dataset LA for METR_LA; BAY for PEMS_BAY",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of epoches for training"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for update parameters",
    )
    args = parser.parse_args()
    # Load the datasets
    if args.dataset == "LA":
        g = METR_LAGraphDataset()
        train_data = METR_LATrainDataset()
        test_data = METR_LATestDataset()
        valid_data = METR_LAValidDataset()
    elif args.dataset == "BAY":
        g = PEMS_BAYGraphDataset()
        train_data = PEMS_BAYTrainDataset()
        test_data = PEMS_BAYTestDataset()
        valid_data = PEMS_BAYValidDataset()

    if args.gpu == -1:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:{}".format(args.gpu))

    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    normalizer = NormalizationLayer(train_data.mean, train_data.std)

    if args.model == "dcrnn":
        batch_g = dgl.batch([g] * args.batch_size).to(device)
        out_gs, in_gs = DiffConv.attach_graph(batch_g, args.diffsteps)
        net = partial(
            DiffConv,
            k=args.diffsteps,
            in_graph_list=in_gs,
            out_graph_list=out_gs,
        )
    elif args.model == "gaan":
        net = partial(GatedGAT, map_feats=64, num_heads=args.num_heads)

    dcrnn = GraphRNN(
        in_feats=2,
        out_feats=64,
        seq_len=12,
        num_layers=2,
        net=net,
        decay_steps=args.decay_steps,
    ).to(device)

    optimizer = torch.optim.Adam(dcrnn.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    loss_fn = masked_mae_loss

    for e in range(args.epochs):
        train_loss = train(
            dcrnn,
            g,
            train_loader,
            optimizer,
            scheduler,
            normalizer,
            loss_fn,
            device,
            args,
        )
        valid_loss = eval(
            dcrnn, g, valid_loader, normalizer, loss_fn, device, args
        )
        test_loss = eval(
            dcrnn, g, test_loader, normalizer, loss_fn, device, args
        )
        print(
            "\rEpoch: {} Train Loss: {} Valid Loss: {} Test Loss: {}".format(
                e, train_loss, valid_loss, test_loss
            )
        )
