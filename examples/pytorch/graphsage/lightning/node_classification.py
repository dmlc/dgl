import glob
import os

import dgl
import dgl.nn.pytorch as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics.functional as MF
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
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

    def inference(self, g, device, batch_size, num_workers, buffer_device=None):
        # The difference between this inference function and the one in the official
        # example is that the intermediate results can also benefit from prefetching.
        g.ndata["h"] = g.ndata["feat"]
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(
            1, prefetch_node_feats=["h"]
        )
        dataloader = dgl.dataloading.DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
        )
        if buffer_device is None:
            buffer_device = device

        for l, layer in enumerate(self.layers):
            y = torch.zeros(
                g.num_nodes(),
                self.n_hidden if l != len(self.layers) - 1 else self.n_classes,
                device=buffer_device,
            )
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = blocks[0].srcdata["h"]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[output_nodes] = h.to(buffer_device)
            g.ndata["h"] = y
        return y

    def training_step(self, batch, batch_idx):
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
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
        input_nodes, output_nodes, blocks = batch
        x = blocks[0].srcdata["feat"]
        y = blocks[-1].dstdata["label"]
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
    def __init__(
        self, graph, train_idx, val_idx, fanouts, batch_size, n_classes
    ):
        super().__init__()

        sampler = dgl.dataloading.NeighborSampler(
            fanouts, prefetch_node_feats=["feat"], prefetch_labels=["label"]
        )

        self.g = graph
        self.train_idx, self.val_idx = train_idx, val_idx
        self.sampler = sampler
        self.batch_size = batch_size
        self.in_feats = graph.ndata["feat"].shape[1]
        self.n_classes = n_classes

    def train_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.train_idx.to("cuda"),
            self.sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            # For CPU sampling, set num_workers to nonzero and use_uva=False
            # Set use_ddp to False for single GPU.
            num_workers=0,
            use_uva=True,
            use_ddp=True,
        )

    def val_dataloader(self):
        return dgl.dataloading.DataLoader(
            self.g,
            self.val_idx.to("cuda"),
            self.sampler,
            device="cuda",
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
            use_uva=True,
        )


if __name__ == "__main__":
    dataset = DglNodePropPredDataset("ogbn-products")
    graph, labels = dataset[0]
    graph.ndata["label"] = labels.squeeze()
    graph.create_formats_()
    split_idx = dataset.get_idx_split()
    train_idx, val_idx, test_idx = (
        split_idx["train"],
        split_idx["valid"],
        split_idx["test"],
    )
    datamodule = DataModule(
        graph, train_idx, val_idx, [15, 10, 5], 1024, dataset.num_classes
    )
    model = SAGE(datamodule.in_feats, 256, datamodule.n_classes)

    # Train
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", save_top_k=1)
    # Use this for single GPU
    # trainer = Trainer(accelerator="gpu", devices=[0], max_epochs=10,
    #                   callbacks=[checkpoint_callback])
    trainer = Trainer(
        accelerator="gpu",
        devices=[0, 1, 2, 3],
        max_epochs=10,
        callbacks=[checkpoint_callback],
        strategy="ddp_spawn",
    )
    trainer.fit(model, datamodule=datamodule)

    # Test
    dirs = glob.glob("./lightning_logs/*")
    version = max([int(os.path.split(x)[-1].split("_")[-1]) for x in dirs])
    logdir = "./lightning_logs/version_%d" % version
    print("Evaluating model in", logdir)
    ckpt = glob.glob(os.path.join(logdir, "checkpoints", "*"))[0]

    model = SAGE.load_from_checkpoint(
        checkpoint_path=ckpt, hparams_file=os.path.join(logdir, "hparams.yaml")
    ).to("cuda")
    with torch.no_grad():
        pred = model.inference(graph, "cuda", 4096, 12, graph.device)
        pred = pred[test_idx]
        label = graph.ndata["label"][test_idx]
        acc = MF.accuracy(
            pred, label, task="multiclass", num_classes=datamodule.n_classes
        )
    print("Test accuracy:", acc)
