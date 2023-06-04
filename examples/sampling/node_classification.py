"""
If you want a deeper understanding of node classification.
You can read the example in the  ``examples/core/graphsage/node_
classification.py``
"""

import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        #####################################################################
        # (HIGHLIGHT) Creating a MultiLayerFullNeighborSampler instance.
        # This sampler is used in the Graph Neural Networks (GNN) training 
        # process to provide neighbor sampling, which is crucial for 
        # efficient training of GNN on large graphs.
        #
        # The first argument '1' indicates the number of layers for 
        # the neighbor sampling. In this case, it's set to 1, meaning 
        # only the direct neighbors of each node will be included in the 
        # sampling.
        #
        # The 'prefetch_node_feats' parameter specifies the node features 
        # that need to be pre-fetched during sampling. In this case, the 
        # feature named 'feat' will be pre-fetched, which should be part 
        # of the nodes' data.
        #####################################################################
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])

        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if is_last_layer else self.out_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if layer_idx != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # By design, our output nodes are contiguous.
                y[output_nodes[0]:output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        pred = model.inference(
            graph, device, batch_size
        )  # pred in buffer_device.
        pred = pred[nid]
        label = graph.ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(args, device, g, dataset, model, num_classes):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)
    #####################################################################
    # (HIGHLIGHT) Instantiate a NeighborSampler object for efficient 
    # training of Graph Neural Networks (GNNs) on large-scale graphs.
    #
    # The argument [10, 10, 10] sets the number of neighbors (fanout) 
    # to be sampled at each layer. Here, we have three layers, and 
    # 10 neighbors will be randomly selected for each node at each 
    # layer.
    #
    # The 'prefetch_node_feats' and 'prefetch_labels' parameters 
    # specify the node features and labels that need to be pre-fetched 
    # during sampling. Pre-fetching data in this way is advantageous 
    # as it can reduce the I/O overhead during the training process, 
    # making the whole computation more efficient. In this case, 
    # the feature 'feat' and the label 'label' will be pre-fetched.
    #####################################################################
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            f"Epoch {epoch:05d} | Loss {total_loss:.4f} | "
            f"Accuracy {acc.item():.4f} "
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed "
        "training, 'puregpu' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # Load and preprocess dataset.
    print("Loading data")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    num_classes = dataset.num_classes
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # Model training.
    print("Training...")
    train(args, device, g, dataset, model, num_classes)

    # Test the model.
    print("Testing...")
    acc = layerwise_infer(
        device, g, dataset.test_idx, model, num_classes, batch_size=4096
    )
    print(f"Test Accuracy {acc.item():.4f}")
