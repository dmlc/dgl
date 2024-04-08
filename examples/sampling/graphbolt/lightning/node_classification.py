"""
This flowchart describes the main functional sequence of the provided example.
main
│
├───> Instantiate DataModule
│     │
│     └───> Load dataset
│     │
│     └───> Create train and valid dataloader[HIGHLIGHT]
│           │
│           └───> ItemSampler (Distribute data to minibatchs)
│           │
│           └───> sample_neighbor or sample_layer_neighbor
                  (Sample a subgraph for a minibatch)
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
└───> Run
      │
      │
      └───> Trainer[HIGHLIGHT]
            │
            ├───> SAGE.forward (GraphSAGE model forward pass)
            │
            └───> Validate
"""
import argparse

import dgl.graphbolt as gb
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
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

    def log_node_and_edge_counts(self, blocks):
        node_counts = [block.num_src_nodes() for block in blocks] + [
            blocks[-1].num_dst_nodes()
        ]
        edge_counts = [block.num_edges() for block in blocks]
        for i, c in enumerate(node_counts):
            self.log(
                f"num_nodes/{i}",
                float(c),
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )
            if i < len(edge_counts):
                self.log(
                    f"num_edges/{i}",
                    float(edge_counts[i]),
                    prog_bar=True,
                    on_step=True,
                    on_epoch=False,
                )

    def training_step(self, batch, batch_idx):
        blocks = [block.to("cuda") for block in batch.blocks]
        x = batch.node_features["feat"]
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
        self.log_node_and_edge_counts(blocks)
        return loss

    def validation_step(self, batch, batch_idx):
        blocks = [block.to("cuda") for block in batch.blocks]
        x = batch.node_features["feat"]
        y = batch.labels.to("cuda")
        y_hat = self(blocks, x)
        self.val_acc(torch.argmax(y_hat, 1), y)
        self.log(
            "val_acc",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log_node_and_edge_counts(blocks)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=5e-4
        )
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, dataset, fanouts, batch_size, num_workers):
        super().__init__()
        self.fanouts = fanouts
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.feature_store = dataset.feature
        self.graph = dataset.graph
        self.train_set = dataset.tasks[0].train_set
        self.valid_set = dataset.tasks[0].validation_set
        self.num_classes = dataset.tasks[0].metadata["num_classes"]

    def create_dataloader(self, node_set, is_train):
        datapipe = gb.ItemSampler(
            node_set,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
        )
        sampler = (
            datapipe.sample_layer_neighbor
            if is_train
            else datapipe.sample_neighbor
        )
        datapipe = sampler(self.graph, self.fanouts)
        datapipe = datapipe.fetch_feature(self.feature_store, ["feat"])
        dataloader = gb.DataLoader(datapipe, num_workers=self.num_workers)
        return dataloader

    ########################################################################
    # (HIGHLIGHT) The 'train_dataloader' and 'val_dataloader' hooks are
    # essential components of the Lightning framework, defining how data is
    # loaded during training and validation. In this example, we utilize a
    # specialized 'graphbolt dataloader', which are concatenated by a series
    # of datapipes, for these purposes.
    ########################################################################
    def train_dataloader(self):
        return self.create_dataloader(self.train_set, is_train=True)

    def val_dataloader(self):
        return self.create_dataloader(self.valid_set, is_train=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GNN baselines on ogbn-products data with GraphBolt"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="number of GPUs used for computing (default: 1)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="input batch size for training (default: 1024)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=40,
        help="number of epochs to train (default: 40)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="number of workers (default: 0)",
    )
    args = parser.parse_args()

    dataset = gb.BuiltinDataset("ogbn-products").load()
    datamodule = DataModule(
        dataset,
        [10, 10, 10],
        args.batch_size,
        args.num_workers,
    )
    in_size = dataset.feature.size("node", None, "feat")[0]
    model = SAGE(in_size, 256, datamodule.num_classes)

    # Train.
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max")
    early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max")
    ########################################################################
    # (HIGHLIGHT) The `Trainer` is the key Class in lightning, which automates
    # everything after defining `LightningDataModule` and
    # `LightningDataModule`. More details can be found in
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html.
    ########################################################################
    trainer = Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stopping_callback],
    )
    trainer.fit(model, datamodule=datamodule)
