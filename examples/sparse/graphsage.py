"""
This script trains and tests a GraphSAGE model based on the information of 
a full graph.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess full dataset
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     └───> Training loop
│           │
│           └───> SAGE.forward
└───> test
      │
      └───> Evaluate the model
"""
import argparse

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
    """

    def __init__(
        self,
        in_feats,
        out_feats,
    ):
        super(SAGEConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_feats, in_feats
        self._out_feats = out_feats

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, A, feat):
        # Remove duplicate edges.
        A = A.coalesce()
        feat_src = feat
        feat_dst = feat[: A.shape[1]]

        # Aggregator type: mean.
        srcdata = self.fc_neigh(feat_src)
        # Divided by degree.
        D_hat = dglsp.diag(A.sum(0)) ** -1
        A_div = A @ D_hat
        # Conv neighbors.
        dstdata = A_div.T @ srcdata

        rst = self.fc_self(feat_dst) + dstdata
        return rst


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-gcn.
        self.layers.append(SAGEConv(in_size, hid_size))
        self.layers.append(SAGEConv(hid_size, hid_size))
        self.layers.append(SAGEConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, sampled_mats, x):
        hidden_x = x
        for layer_idx, (layer, sampled_mat) in enumerate(
            zip(self.layers, sampled_mats)
        ):
            hidden_x = layer(sampled_mat, hidden_x)
            if layer_idx != len(self.layers) - 1:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x

    def inference(self, A, dataset, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = dataset[0].ndata["feat"]
        inf_idx = dataset.val_idx.to(device)
        inf_dataloader = torch.utils.data.DataLoader(
            inf_idx, batch_size=batch_size
        )

        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        node_num = A.shape[0]
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                node_num,
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)

            for it, (seeds) in enumerate(inf_dataloader):
                # Sampling all neighbors.
                sampled_mat = A.sample(1, fanout=node_num, ids=seeds)
                # Compact the matrix.
                compacted_mat, row_ids = sampled_mat.compact(0)
                x = feat[row_ids]
                h = layer(compacted_mat, x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                y[seeds] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, dataloader, g, num_classes):
    model.eval()
    ys = []
    y_hats = []
    fanouts = [10, 10, 10]
    for it, (seeds) in enumerate(dataloader):
        with torch.no_grad():
            src = seeds
            sampled_mats = []
            for fanout in fanouts:
                # Sampling neighbor
                sampled_mat = A.sample(1, fanout, ids=src, replace=True)
                # Compact the matrix
                compacted_mat, row_ids = sampled_mat.compact(0)
                sampled_mats.insert(0, compacted_mat)
                src = row_ids

            x = g.ndata["feat"][src]
            y = g.ndata["label"][seeds]
            ys.append(y)
            y_hats.append(model(sampled_mats, x))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, A, dataset, model, num_classes, batch_size):
    model.eval()
    nid = dataset.test_idx
    with torch.no_grad():
        pred = model.inference(
            A, dataset, device, batch_size
        )  # pred in buffer_device
        pred = pred[nid]
        label = dataset[0].ndata["label"][nid].to(pred.device)
        return MF.accuracy(
            pred, label, task="multiclass", num_classes=num_classes
        )


def train(device, A, g, dataset, model, num_classes):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_idx, batch_size=1024)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    fanouts = [10, 10, 10]
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (seeds) in enumerate(train_dataloader):
            src = seeds
            sampled_mats = []
            for fanout in fanouts:
                # Sampling neighbor
                sampled_mat = A.sample(1, fanout, ids=src, replace=True)
                # Compact the matrix
                compacted_mat, row_ids = sampled_mat.compact(0)
                sampled_mats.insert(0, compacted_mat)
                src = row_ids

            x = g.ndata["feat"][src]
            y = g.ndata["label"][seeds]
            y_hat = model(sampled_mats, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        acc = evaluate(model, val_dataloader, g, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--mode",
        default="gpu",
        choices=["cpu", "gpu"],
        help="Training mode. 'cpu' for CPU training, "
        "'gpu' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    #####################################################################
    # (HIGHLIGHT) Node classification task is a supervise learning task
    # in which the model try to predict the label of a certain node.
    # In this example, graph sage algorithm is applied to this task.
    # A good accuracy can be achieved after a few steps of training.
    #
    # First, the whole graph is loaded and transformed. Then the training
    # process is performed on a model which is composed of 3 GraphSAGE-gcn
    # layers. Finally, the performance of the model is evaluated on test set.
    #####################################################################

    # Load and preprocess dataset.
    print("Loading data")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to(device)
    num_classes = dataset.num_classes

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    out_size = dataset.num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # Create sparse.
    indices = torch.stack(g.edges())
    N = g.num_nodes()
    A = dglsp.spmatrix(indices, shape=(N, N))

    # Model training.
    print("Training...")
    train(device, A, g, dataset, model, num_classes)

    # Test the model.
    print("Testing...")
    acc = layerwise_infer(
        device, A, dataset, model, num_classes, batch_size=4096
    )
    print(f"Test accuracy {acc:.4f}")
