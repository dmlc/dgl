"""
This flowchart describes the main functional sequence of the provided example.
main
│
├───> initialize DataModule
│     │
│     └───> Load dataset
│     │
│     └───> Create train and valid dataloader[HIGHLIGHT]
│           │
│           └───> ItemSampler (Distribute data to minibatchs)
│           │
│           └───> sample_neighbor (Sample a subgraph for a minibatch)
│           │
│           └───> fetch_feature (Fetch features for the sampled subgraph)
│
├───> Instantiate GraphSAGE model
│     │
│     ├───> SAGEConvLayer (input to hidden)
│     │
│     └───> SAGEConvLayer (hidden to hidden)
│     │
│     └───> SAGEConvLayer (hidden to output)
│     │
│     └───> DropoutLayer
│
└───> run
      │
      │
      └───> Trainer[HIGHLIGHT]
            │
            ├───> SAGE.forward (RGCN model forward pass)
            │
            └───> validate
"""
import argparse

import dgl.graphbolt as gb
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import Accuracy


class SAGE(LightningModule):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.train_acc = Accuracy(task="multiclass", num_classes=n_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=n_classes)

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def training_step(self, batch, batch_idx):
        blocks = [block.to("cuda") for block in batch.to_dgl_blocks()]
        x = blocks[0].srcdata["feat"]
        y = batch.labels.to("cuda")
        y_hat = self(blocks, x)
        loss = F.cross_entropy(y_hat, y)
        self.train_acc(torch.argmax(y_hat, 1), y)
        self.log(
            "train_acc",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        blocks = [block.to("cuda") for block in batch.to_dgl_blocks()]
        x = blocks[0].srcdata["feat"]
        y = batch.labels.to("cuda")
        y_hat = self(blocks, x)
        self.val_acc(torch.argmax(y_hat, 1), y)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=5e-4
        )
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, fanouts, batch_size, num_workers):
        super().__init__()
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.num_workers = num_workers
        dataset = gb.OnDiskDataset(
            "/home/ubuntu/workspace/example_ogbn_products/"
        )
        dataset.load()
        self.feature_store = dataset.feature
        self.graph = dataset.graph
        self.train_set = dataset.tasks[0].train_set
        self.valid_set = dataset.tasks[0].validation_set
        self.num_classes = dataset.tasks[0].metadata["num_classes"]

    ########################################################################
    # (HIGHLIGHT) The 'train_dataloader' and 'val_dataloader' hooks are
    # essential components of the Lightning framework, defining how data is
    # loaded during training and validation. In this example, we utilize a
    # specialized 'graphbolt dataloader', which are cconcatenated by a series
    # of datappipes, for these purposes.
    ########################################################################
    def train_dataloader(self):
        datapipe = gb.ItemSampler(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        datapipe = datapipe.sample_neighbor(self.graph, self.fanouts)
        datapipe = datapipe.fetch_feature(self.feature_store, ["feat"])
        dataloader = gb.MultiProcessDataLoader(
            datapipe, num_workers=self.num_workers
        )
        return dataloader

    def val_dataloader(self):
        datapipe = gb.ItemSampler(
            self.valid_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        datapipe = datapipe.sample_neighbor(self.graph, self.fanouts)
        datapipe = datapipe.fetch_feature(self.feature_store, ["feat"])
        dataloader = gb.MultiProcessDataLoader(
            datapipe, num_workers=self.num_workers
        )
        return dataloader


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbgmol* data with Pytorch Geometrics"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=4,
        help="number of GPUs used for computing (default: 4)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    args = parser.parse_args()

    datamodule = DataModule([15, 10, 5], args.batch_size, args.num_workers)
    model = SAGE(100, 256, datamodule.num_classes).to(torch.double)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", save_top_k=1)
    ########################################################################
    # (HIGHLIGHT) The `Trainer` is the key Class in lightning, which automates
    # everything for you after defining `LightningDataModule` and
    # `LightningDataModule`. More details can be found in
    # https://github.com/dmlc/dgl/pull/6335.
    ########################################################################
    trainer = Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)
