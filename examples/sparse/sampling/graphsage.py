"""
This script demonstrate how to use dgl sparse library to sample on graph and 
train model. It trains and tests a GraphSAGE model using the sparse sample and 
compact operators to sample submatrix from the whole matrix.

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
│           ├───> Sample submatrix
│           │
│           └───> SAGE.forward
└───> test
      │
      ├───> Sample submatrix
      │
      └───> Evaluate the model
"""
import argparse

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.data import AsNodePredDataset
from ogb.nodeproppred import DglNodePropPredDataset


class SAGEConv(nn.Module):
    r"""GraphSAGE layer from `Inductive Representation Learning on
    Large Graphs <https://arxiv.org/pdf/1706.02216.pdf>`__
    """

    def __init__(
        self,
        in_size,
        out_size,
    ):
        super(SAGEConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = in_size, in_size
        self._out_size = out_size

        self.fc_neigh = nn.Linear(self._in_src_feats, out_size, bias=False)
        self.fc_self = nn.Linear(self._in_dst_feats, out_size, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, A, feat):
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

    def forward(self, sampled_matrices, x):
        hidden_x = x
        for layer_idx, (layer, sampled_matrix) in enumerate(
            zip(self.layers, sampled_matrices)
        ):
            hidden_x = layer(sampled_matrix, hidden_x)
            if layer_idx != len(self.layers) - 1:
                hidden_x = F.relu(hidden_x)
                hidden_x = self.dropout(hidden_x)
        return hidden_x


def multilayer_sample(A, fanouts, seeds, ndata):
    sampled_matrices = []
    src = seeds

    #####################################################################
    # (HIGHLIGHT) Using the sparse sample operator to preform random
    # sampling on the neighboring nodes of the seeds nodes. The sparse
    # compact operator is then employed to compact and relabel the sampled
    # matrix, resulting in the sampled matrix and the relabel index.
    #####################################################################

    for fanout in fanouts:
        # Sample neighbors.
        sampled_matrix = A.sample(1, fanout, ids=src).coalesce()
        # Compact the sampled matrix.
        compacted_mat, row_ids = sampled_matrix.compact(0)
        sampled_matrices.insert(0, compacted_mat)
        src = row_ids

    x = ndata["feat"][src]
    y = ndata["label"][seeds]
    return sampled_matrices, x, y


def evaluate(model, A, dataloader, ndata, num_classes):
    model.eval()
    ys = []
    y_hats = []
    fanouts = [10, 10, 10]
    for it, seeds in enumerate(dataloader):
        with torch.no_grad():
            sampled_matrices, x, y = multilayer_sample(A, fanouts, seeds, ndata)
            ys.append(y)
            y_hats.append(model(sampled_matrices, x))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def validate(device, A, ndata, dataset, model, batch_size):
    inf_id = dataset.test_idx.to(device)
    inf_dataloader = torch.utils.data.DataLoader(inf_id, batch_size=batch_size)
    acc = evaluate(model, A, inf_dataloader, ndata, dataset.num_classes)
    return acc


def train(device, A, ndata, dataset, model):
    # Create sampler & dataloader.
    train_idx = dataset.train_idx.to(device)
    val_idx = dataset.val_idx.to(device)

    train_dataloader = torch.utils.data.DataLoader(
        train_idx, batch_size=1024, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val_idx, batch_size=1024)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    fanouts = [10, 10, 10]
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, seeds in enumerate(train_dataloader):
            sampled_matrices, x, y = multilayer_sample(A, fanouts, seeds, ndata)
            y_hat = model(sampled_matrices, x)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, A, val_dataloader, ndata, dataset.num_classes)
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
        help="Training mode. 'cpu' for CPU training, 'gpu' for GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    #####################################################################
    # (HIGHLIGHT) This example implements a graphSAGE algorithm by sparse
    # operators, which involves sampling a subgraph from a full graph and
    # conducting training.
    #
    # First, the whole graph is loaded onto the CPU or GPU and transformed
    # to sparse matrix. To obtain the training subgraph, it samples three
    # submatrices by seed nodes, which contains their randomly sampled
    # 1-hop, 2-hop, and 3-hop neighbors. Then, the features of the
    # subgraph are input to the network for training.
    #####################################################################

    # Load and preprocess dataset.
    print("Loading data")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    dataset = AsNodePredDataset(DglNodePropPredDataset("ogbn-products"))
    g = dataset[0]
    g = g.to(device)

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
    train(device, A, g.ndata, dataset, model)

    # Test the model.
    print("Testing...")
    acc = validate(device, A, g.ndata, dataset, model, batch_size=4096)
    print(f"Test accuracy {acc:.4f}")
