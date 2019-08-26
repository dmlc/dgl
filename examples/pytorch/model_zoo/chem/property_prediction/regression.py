# -*- coding:utf-8 -*-
"""Sample training code
"""

import argparse
import torch as th
import torch.nn as nn
from dgl.data import alchemy
from dgl import model_zoo
from torch.utils.data import DataLoader
# from Alchemy_dataset import TencentAlchemyDataset, batcher


def train(model="sch",
          epochs=80,
          device=th.device("cpu"),
          training_set_size=0.8):
    print("start")
    alchemy_dataset = alchemy.TencentAlchemyDataset()
    train_set, test_set = alchemy_dataset.split(train_size=0.8)
    train_loader = DataLoader(dataset=train_set,
                              batch_size=20,
                              collate_fn=alchemy.batcher(),
                              shuffle=False,
                              num_workers=0)
    test_loader = DataLoader(dataset=test_set,
                             batch_size=20,
                             collate_fn=alchemy.batcher(),
                             shuffle=False,
                             num_workers=0)

    if model == "sch":
        model = model_zoo.chem.SchNetModel(norm=True, output_dim=12)
        model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    elif model == "mgcn":
        model = model_zoo.chem.MGCNModel(norm=True, output_dim=12)
        model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    elif model == "mpnn":
        model = model_zoo.chem.MPNNModel(output_dim=12)

    model.to(device)

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(train_loader):
            batch.graph.to(device)
            batch.label = batch.label.to(device)

            res = model(batch.graph)
            loss = loss_fn(res, batch.label)
            mae = MAE_fn(res, batch.label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            w_mae += mae.detach().item()
            w_loss += loss.detach().item()
        w_mae /= idx + 1

        print("Epoch {:2d}, loss: {:.7f}, MAE: {:.7f}".format(
            epoch, w_loss, w_mae))

    w_loss, w_mae = 0, 0
    model.eval()

    for idx, batch in enumerate(test_loader):
        batch.graph.to(device)
        batch.label = batch.label.to(device)

        res = model(batch.graph)
        mae = MAE_fn(res, batch.label)

        w_mae += mae.detach().item()
        w_loss += loss.detach().item()
    w_mae /= idx + 1
    print("MAE (test set): {:.7f}".format(w_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch, mgcn, mpnn)",
                        choices=['sch', 'mgcn', 'mpnn'],
                        default="sch")
    parser.add_argument("--epochs",
                        help="number of epochs",
                        default=250,
                        type=int)
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    assert args.model in ['sch', 'mgcn',
                          'mpnn'], "model name must be sch, mgcn or mpnn"
    train(args.model, args.epochs, device)
