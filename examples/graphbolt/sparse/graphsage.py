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
from functools import partial

import dgl.graphbolt as gb

import dgl.sparse as dglsp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from dgl.graphbolt.subgraph_sampler import SubgraphSampler
from torch.utils.data import functional_datapipe
from tqdm import tqdm


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


@functional_datapipe("sample_sparse_neighbor")
class SparseNeighborSampler(SubgraphSampler):
    def __init__(self, datapipe, matrix, fanouts):
        super().__init__(datapipe)
        self.matrix = matrix
        # Convert fanouts to a list of tensors.
        self.fanouts = []
        for fanout in fanouts:
            if not isinstance(fanout, torch.Tensor):
                fanout = torch.LongTensor([int(fanout)])
            self.fanouts.insert(0, fanout)

    def sample_subgraphs(self, seeds, seeds_timestamp=None):
        sampled_matrices = []
        src = seeds.long()

        #####################################################################
        # (HIGHLIGHT) Using the sparse sample operator to preform random
        # sampling on the neighboring nodes of the seeds nodes. The sparse
        # compact operator is then employed to compact and relabel the sampled
        # matrix, resulting in the sampled matrix and the relabel index.
        #####################################################################
        for fanout in self.fanouts:
            # Sample neighbors.
            sampled_matrix = self.matrix.sample(1, fanout, ids=src).coalesce()
            # Compact the sampled matrix.
            compacted_mat, row_ids = sampled_matrix.compact(0)
            sampled_matrices.insert(0, compacted_mat)
            src = row_ids

        return src, sampled_matrices


############################################################################
# (HIGHLIGHT) Create a multi-process dataloader with dgl graphbolt package.
############################################################################
def create_dataloader(A, fanouts, ids, features, device):
    datapipe = gb.ItemSampler(ids, batch_size=1024)
    # Customize graphbolt sampler by sparse.
    datapipe = datapipe.sample_sparse_neighbor(A, fanouts)
    # Use grapbolt to fetch features.
    datapipe = datapipe.fetch_feature(features, node_feature_keys=["feat"])
    datapipe = datapipe.copy_to(device)
    dataloader = gb.DataLoader(datapipe)
    return dataloader


def evaluate(model, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, data in tqdm(enumerate(dataloader), "Evaluating"):
        with torch.no_grad():
            node_feature = data.node_features["feat"].float()
            blocks = data.sampled_subgraphs
            y = data.labels
            ys.append(y)
            y_hats.append(model(blocks, node_feature))

    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def validate(device, dataset, model, num_classes):
    test_set = dataset.tasks[0].test_set
    test_dataloader = create_dataloader(
        A, [10, 10, 10], test_set, features, device
    )
    acc = evaluate(model, test_dataloader, num_classes)
    return acc


def train(device, A, features, dataset, num_classes, model):
    # Create sampler & dataloader.
    train_set = dataset.tasks[0].train_set
    train_dataloader = create_dataloader(
        A, [10, 10, 10], train_set, features, device
    )

    valid_set = dataset.tasks[0].validation_set
    val_dataloader = create_dataloader(
        A, [10, 10, 10], valid_set, features, device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, data in tqdm(enumerate(train_dataloader), "Training"):
            node_feature = data.node_features["feat"].float()
            blocks = data.sampled_subgraphs
            y = data.labels
            y_hat = model(blocks, node_feature)
            loss = F.cross_entropy(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, val_dataloader, num_classes)
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
    dataset = gb.BuiltinDataset("ogbn-products").load()
    g = dataset.graph
    features = dataset.feature

    # Create GraphSAGE model.
    in_size = features.size("node", None, "feat")[0]
    num_classes = dataset.tasks[0].metadata["num_classes"]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # Create sparse.
    N = g.num_nodes
    A = dglsp.from_csc(g.csc_indptr.long(), g.indices.long(), shape=(N, N))

    # Model training.
    print("Training...")
    train(device, A, features, dataset, num_classes, model)

    # Test the model.
    print("Testing...")
    acc = validate(device, dataset, model, num_classes)
    print(f"Test accuracy {acc:.4f}")
