import argparse
import glob
import os
import time

import dgl.graphbolt as gb
import torch

# Needed until https://github.com/pytorch/pytorch/issues/121197 is resolved to
# use the `--torch-compile` cmdline option reliably.
import torch._inductor.codecache
import torch.nn as nn
import torch.nn.functional as F

from load_dataset import load_dataset
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from sage_conv import SAGEConv

from torchmetrics.classification import MulticlassF1Score, MultilabelF1Score
from tqdm import tqdm


def convert_to_pyg(h, subgraph):
    #####################################################################
    # (HIGHLIGHT) Convert given features to be consumed by a PyG layer.
    #
    #   We convert the provided sampled edges in CSC format from GraphBolt and
    #   convert to COO via using gb.expand_indptr.
    #####################################################################
    src = subgraph.sampled_csc.indices
    dst = gb.expand_indptr(
        subgraph.sampled_csc.indptr,
        dtype=src.dtype,
        output_size=src.size(0),
    )
    edge_index = torch.stack([src, dst], dim=0).long()
    dst_size = subgraph.sampled_csc.indptr.size(0) - 1
    # h and h[:dst_size] correspond to source and destination features resp.
    return (h, h[:dst_size]), edge_index, (h.size(0), dst_size)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_size, hidden_size, out_size, n_layers, dropout):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        sizes = [in_size] + [hidden_size] * n_layers
        for i in range(n_layers):
            self.layers.append(SAGEConv(sizes[i], sizes[i + 1]))
        self.linear = nn.Linear(hidden_size, out_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size
        self.out_size = out_size

    def forward(self, subgraphs, x):
        h = x
        for layer, subgraph in zip(self.layers, subgraphs):
            h, edge_index, size = convert_to_pyg(h, subgraph)
            h = layer(h, edge_index, size=size)
            h = F.gelu(h)
            h = self.dropout(h)
        return self.linear(h)

    def inference(self, graph, features, dataloader, storage_device):
        """Conduct layer-wise inference to get all the node embeddings."""
        pin_memory = storage_device == "pinned"
        buffer_device = torch.device("cpu" if pin_memory else storage_device)

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1

            y = torch.empty(
                graph.total_num_nodes,
                self.out_size if is_last_layer else self.hidden_size,
                dtype=torch.float32,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            for data in tqdm(dataloader, "Inferencing"):
                # len(data.sampled_subgraphs) = 1
                h, edge_index, size = convert_to_pyg(
                    data.node_features["feat"], data.sampled_subgraphs[0]
                )
                hidden_x = layer(h, edge_index, size=size)
                hidden_x = F.gelu(hidden_x)
                if is_last_layer:
                    hidden_x = self.linear(hidden_x)
                # By design, our output nodes are contiguous.
                y[data.seeds[0] : data.seeds[-1] + 1] = hidden_x.to(
                    buffer_device
                )
            if not is_last_layer:
                features.update("node", None, "feat", y)

        return y


class SAGELightning(LightningModule):
    def __init__(
        self,
        in_feats,
        n_hidden,
        n_classes,
        n_layers,
        lr,
        dropout,
        multilabel,
        torch_compile,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.module = GraphSAGE(
            in_feats, n_hidden, n_classes, n_layers, dropout
        )
        if torch_compile:
            torch._dynamo.config.cache_size_limit = 32
            self.module = torch.compile(
                self.module, fullgraph=True, dynamic=True
            )
        self.lr = lr
        self.f1score_class = lambda: (
            MultilabelF1Score if multilabel else MulticlassF1Score
        )(n_classes, average="micro")
        self.train_acc = self.f1score_class()
        self.val_acc = self.f1score_class()
        self.loss_fn = (
            nn.BCEWithLogitsLoss() if multilabel else nn.CrossEntropyLoss()
        )
        self.multilabel = multilabel
        self.pt = 0

    def forward(self, subgraphs, x):
        return self.module(subgraphs, x)

    def inference(self, graph, features, dataloader, storage_device):
        return self.module.inference(
            graph, features, dataloader, storage_device
        )

    def log_node_and_edge_counts(self, subgraphs):
        node_counts = [sg.original_row_node_ids.size(0) for sg in subgraphs] + [
            subgraphs[-1].original_column_node_ids.size(0)
        ]
        edge_counts = [sg.sampled_csc.indices.size(0) for sg in subgraphs]
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

    def training_step(self, minibatch, batch_idx):
        batch_inputs = minibatch.node_features["feat"]
        batch_labels = minibatch.labels
        self.st = time.time()
        batch_pred = self(minibatch.sampled_subgraphs, batch_inputs)
        label_dtype = batch_pred.dtype if self.multilabel else None
        loss = self.loss_fn(batch_pred, batch_labels.to(label_dtype))
        self.train_acc(batch_pred, batch_labels.int())
        self.log(
            "acc/train",
            self.train_acc,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch_labels.shape[0],
        )
        self.log(
            "loss/train",
            loss,
            on_step=True,
            on_epoch=True,
            batch_size=batch_labels.shape[0],
        )
        t = time.time()
        self.log(
            "time/iter",
            t - self.pt,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )
        self.log_node_and_edge_counts(minibatch.sampled_subgraphs)
        self.pt = t
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log(
            "time/forward_backward",
            time.time() - self.st,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
        )

    def validation_step(self, minibatch, batch_idx, dataloader_idx=0):
        batch_inputs = minibatch.node_features["feat"]
        batch_labels = minibatch.labels
        batch_pred = self(minibatch.sampled_subgraphs, batch_inputs)
        label_dtype = batch_pred.dtype if self.multilabel else None
        loss = self.loss_fn(batch_pred, batch_labels.to(label_dtype))
        self.val_acc(batch_pred, batch_labels.int())
        self.log(
            "acc/val",
            self.val_acc,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_labels.shape[0],
        )
        self.log(
            "loss/val",
            loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_labels.shape[0],
        )
        self.log_node_and_edge_counts(minibatch.sampled_subgraphs)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


class DataModule(LightningDataModule):
    def __init__(self, args):
        super().__init__()

        dataset, multilabel = load_dataset(args.dataset)

        # Move the dataset to the selected storage.
        graph = (
            dataset.graph.pin_memory_()
            if args.graph_device == "pinned"
            else dataset.graph.to(args.graph_device)
        )
        features = (
            dataset.feature.pin_memory_()
            if args.feature_device == "pinned"
            else dataset.feature.to(args.feature_device)
        )

        self.train_set = dataset.tasks[0].train_set
        self.validation_set = dataset.tasks[0].validation_set
        self.test_set = dataset.tasks[0].test_set
        self.all_nodes_set = dataset.all_nodes_set
        self.fanout = list(map(int, args.fanout.split(",")))

        if args.num_gpu_cached_features > 0 and args.feature_device != "cuda":
            feature = features._features[("node", None, "feat")]
            features._features[("node", None, "feat")] = gb.GPUCachedFeature(
                feature,
                args.num_gpu_cached_features * feature._tensor[:1].nbytes,
            )

        self.graph = graph
        self.features = features
        self.feature_device = args.feature_device
        self.batch_size = args.batch_size
        self.num_workers = args.num_workers
        self.in_feats = features.size("node", None, "feat")[0]
        self.n_classes = dataset.tasks[0].metadata["num_classes"]
        self.multilabel = multilabel
        self.device = args.device

    def create_dataloader(self, itemset, job):
        # Initialize an ItemSampler to sample mini-batches from the dataset.
        datapipe = gb.ItemSampler(
            itemset,
            batch_size=self.batch_size * (1 if job != "infer" else 4),
            shuffle=(job == "train"),
            drop_last=(job == "train"),
        )
        # Copy the data to the specified device.
        if args.graph_device != "cpu":
            datapipe = datapipe.copy_to(device=self.device)
        sample_mode = args.sample_mode if job == "train" else "sample_neighbor"
        # Sample neighbors for each node in the mini-batch.
        kwargs = (
            {
                "layer_dependency": args.layer_dependency,
                "batch_dependency": args.batch_dependency,
            }
            if sample_mode == "sample_layer_neighbor"
            else {}
        )
        datapipe = getattr(datapipe, sample_mode)(
            self.graph, self.fanout if job != "infer" else [-1], **kwargs
        )
        # Copy the data to the specified device.
        if args.feature_device != "cpu":
            datapipe = datapipe.copy_to(device=self.device)
        # Fetch node features for the sampled subgraph.
        datapipe = datapipe.fetch_feature(
            self.features, node_feature_keys=["feat"]
        )
        # Copy the data to the specified device.
        if args.feature_device == "cpu":
            datapipe = datapipe.copy_to(device=self.device)
        # Create and return a DataLoader to handle data loading.
        return gb.DataLoader(
            datapipe,
            num_workers=args.num_workers,
            overlap_graph_fetch=args.overlap_graph_fetch,
        )

    def train_dataloader(self):
        return self.create_dataloader(self.train_set, "train")

    def val_dataloader(self):
        return self.create_dataloader(self.validation_set, "evaluate")

    @torch.no_grad()
    def layerwise_infer(self, model, logger):
        model.eval()
        dataloader = self.create_dataloader(
            itemset=self.all_nodes_set,
            job="infer",
        )
        pred = model.inference(
            self.graph, self.features, dataloader, self.feature_device
        )

        metrics = {}
        for itemset, split_name in zip(
            [
                self.train_set,
                self.validation_set,
                self.test_set,
            ],
            ["train", "val", "test"],
        ):
            nid, labels = itemset[:]
            f1score = model.f1score_class().to(pred.device)
            acc = f1score(pred[nid.to(pred.device)], labels.to(pred.device))
            print(f"{split_name} accuracy: {acc.item()}")
            metrics["acc/final_{}".format(split_name)] = acc
        logger.log_metrics(metrics=metrics, step=0)


class CacheMissRateReporterCallback(Callback):
    def report_cache_miss_rate(self, trainer):
        feature = trainer.datamodule.features._features[("node", None, "feat")]
        if isinstance(feature, gb.GPUCachedFeature):
            trainer.strategy.model.log(
                "cache_miss",
                feature._feature.miss_rate,
                prog_bar=True,
                on_step=True,
                on_epoch=False,
            )

    def on_train_batch_end(
        self, trainer, datamodule, outputs, batch, batch_idx
    ):
        self.report_cache_miss_rate(trainer)

    def on_validation_batch_end(
        self, trainer, datamodule, outputs, batch, batch_idx
    ):
        self.report_cache_miss_rate(trainer)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--mode",
        default="pinned-pinned-cuda",
        choices=[
            "cpu-cpu-cpu",
            "cpu-cpu-cuda",
            "cpu-pinned-cuda",
            "pinned-pinned-cuda",
            "cuda-pinned-cuda",
            "cuda-cuda-cuda",
        ],
        help="Graph storage - feature storage - Train device: 'cpu' for CPU and RAM,"
        " 'pinned' for pinned memory in RAM, 'cuda' for GPU and GPU memory.",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-products",
        choices=[
            "ogbn-arxiv",
            "ogbn-products",
            "ogbn-papers100M",
            "reddit",
            "yelp",
            "flickr",
        ],
    )
    argparser.add_argument("--num-epochs", type=int, default=-1)
    argparser.add_argument("--num-steps", type=int, default=-1)
    argparser.add_argument("--min-steps", type=int, default=0)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--fanout", type=str, default="10,10,10")
    argparser.add_argument("--batch-size", type=int, default=1024)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument(
        "--sample-mode",
        default="sample_layer_neighbor",
        choices=["sample_neighbor", "sample_layer_neighbor"],
        help="The sampling function when doing layerwise sampling.",
    )
    argparser.add_argument(
        "--overlap-graph-fetch",
        action="store_true",
        help="An option for enabling overlap_graph_fetch in graphbolt dataloader."
        "If True, the data loader will overlap the UVA graph fetching operations"
        "with the rest of operations by using an alternative CUDA stream. Disabled"
        "by default.",
    )
    argparser.add_argument(
        "--torch-compile",
        action="store_true",
        help="Uses torch.compile() on the trained GNN model. Requires "
        "torch>=2.2.0 to enable this option.",
    )
    argparser.add_argument("--layer-dependency", action="store_true")
    argparser.add_argument("--batch-dependency", type=int, default=1)
    argparser.add_argument("--logdir", type=str, default="tb_logs")
    argparser.add_argument(
        "--num-gpu-cached-features",
        type=int,
        default=0,
        help="The capacity of the GPU cache, the number of features to store.",
    )
    argparser.add_argument("--early-stopping-patience", type=int, default=25)
    argparser.add_argument("--disable-logging", action="store_true")
    argparser.add_argument("--disable-checkpoint", action="store_true")
    argparser.add_argument("--precision", type=str, default="high")
    args = argparser.parse_args()

    torch.set_float32_matmul_precision(args.precision)

    if not torch.cuda.is_available():
        args.mode = "cpu-cpu-cpu"
    print(f"Training in {args.mode} mode.")
    args.graph_device, args.feature_device, args.device = args.mode.split("-")

    num_layers = len(args.fanout.split(","))

    datamodule = DataModule(args)
    model = SAGELightning(
        datamodule.in_feats,
        args.num_hidden,
        datamodule.n_classes,
        num_layers,
        args.lr,
        args.dropout,
        datamodule.multilabel,
        args.torch_compile,
    )

    # Train
    callbacks = []
    if not args.disable_checkpoint:
        callbacks.append(
            ModelCheckpoint(monitor="acc/val", save_top_k=1, mode="max")
        )
    callbacks.append(CacheMissRateReporterCallback())
    callbacks.append(
        EarlyStopping(
            monitor="acc/val",
            mode="max",
            patience=args.early_stopping_patience,
        )
    )
    subdir = "{}_{}_{}_{}_{}".format(
        args.dataset,
        {"sample_layer_neighbor": "labor", "sample_neighbor": "neighbor"}[
            args.sample_mode
        ],
        0,
        args.layer_dependency,
        args.batch_dependency,
    )
    logger = (
        None
        if args.disable_logging
        else TensorBoardLogger(args.logdir, name=subdir)
    )
    trainer = Trainer(
        accelerator="gpu" if args.device == "cuda" else "cpu",
        devices=1,
        max_epochs=args.num_epochs,
        max_steps=args.num_steps,
        min_steps=args.min_steps,
        callbacks=callbacks,
        logger=logger,
    )
    trainer.fit(model, datamodule=datamodule)

    # Test
    if not args.disable_checkpoint:
        logdir = os.path.join(args.logdir, subdir)
        dirs = glob.glob("./{}/*".format(logdir))
        version = max([int(os.path.split(x)[-1].split("_")[-1]) for x in dirs])
        logdir = "./{}/version_{}".format(logdir, version)
        print("Evaluating model in", logdir)
        ckpt = glob.glob(os.path.join(logdir, "checkpoints", "*"))[0]

        model = SAGELightning.load_from_checkpoint(
            checkpoint_path=ckpt,
            hparams_file=os.path.join(logdir, "hparams.yaml"),
        ).to(args.device)

        datamodule.layerwise_infer(model, logger)
