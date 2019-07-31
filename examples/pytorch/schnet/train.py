# -*- coding:utf-8 -*-
"""Sample training code
"""

import argparse
import torch as th
import torch.nn as nn
from sch import SchNetModel
from mgcn import MGCNModel
from torch.utils.data import DataLoader
from Alchemy_dataset import TencentAlchemyDataset, batcher


def train(model="sch", epochs=80, device=th.device("cpu")):
    print("start")
    alchemy_dataset = TencentAlchemyDataset()
    alchemy_loader = DataLoader(dataset=alchemy_dataset,
                                batch_size=20,
                                collate_fn=batcher(),
                                shuffle=False,
                                num_workers=0)

    if model == "sch":
        model = SchNetModel(norm=True, output_dim=12)
    elif model == "mgcn":
        model = MGCNModel(norm=True, output_dim=12)

    model.set_mean_std(alchemy_dataset.mean, alchemy_dataset.std, device)
    model.to(device)

    loss_fn = nn.MSELoss()
    MAE_fn = nn.L1Loss()
    optimizer = th.optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(epochs):

        w_loss, w_mae = 0, 0
        model.train()

        for idx, batch in enumerate(alchemy_loader):
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

        print("Epoch {:2d}, loss: {:.7f}, mae: {:.7f}".format(
            epoch, w_loss, w_mae))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M",
                        "--model",
                        help="model name (sch or mgcn)",
                        default="sch")
    parser.add_argument("--epochs", help="number of epochs", default=250)
    device = th.device('cuda:0' if th.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    train(args.model, int(args.epochs), device)
