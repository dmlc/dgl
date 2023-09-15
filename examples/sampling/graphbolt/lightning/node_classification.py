import dgl
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
        blocks = [block.to('cuda') for block in batch.to_dgl_blocks()]
        x = blocks[0].srcdata["feat"]
        y = batch.labels.to('cuda')
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
        blocks = [block.to('cuda') for block in batch.to_dgl_blocks()]
        x = blocks[0].srcdata["feat"]
        y = batch.labels.to('cuda')
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
    def __init__(self, fanouts, batch_size):
        super().__init__()
        self.fanouts = fanouts
        self.batch_size = batch_size
        dataset = gb.OnDiskDataset(
            "/home/ubuntu/workspace/example_ogbn_products/")
        dataset.load()
        self.feature_store = dataset.feature
        self.graph = dataset.graph
        self.train_set = dataset.tasks[0].train_set
        self.test_set = dataset.tasks[0].test_set
        self.valid_set = dataset.tasks[0].validation_set
        self.num_classes = dataset.tasks[0].metadata["num_classes"]

    def train_dataloader(self):
        item_sampler = gb.ItemSampler(self.train_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=False)
        sampler_dp = gb.NeighborSampler(item_sampler, self.graph, self.fanouts)
        feature_dp = gb.FeatureFetcher(sampler_dp,
                                       self.feature_store, ["feat"])
        dataloader = dgl.graphbolt.SingleProcessDataLoader(feature_dp)
        return dataloader

    def val_dataloader(self):
        item_sampler = gb.ItemSampler(self.valid_set,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      drop_last=False)
        sampler_dp = gb.NeighborSampler(item_sampler, self.graph, self.fanouts)
        feature_dp = gb.FeatureFetcher(sampler_dp,
                                       self.feature_store,
                                       ["feat"])
        dataloader = dgl.graphbolt.SingleProcessDataLoader(feature_dp)
        return dataloader

if __name__ == "__main__":
    datamodule = DataModule([15, 10, 5], 1024)
    model = SAGE(100, 256, datamodule.num_classes).to(torch.double)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", save_top_k=1)
    # Use this for single GPU
    # trainer = Trainer(accelerator="gpu", devices=[0], max_epochs=10,
    #                   callbacks=[checkpoint_callback])
    trainer = Trainer(
        accelerator="gpu",
        devices=8,
        max_epochs=10,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(model, datamodule=datamodule)
