"""
This script is a PyG counterpart of ``/examples/graphbolt/rgcn/hetero_rgcn.py``.
"""

import argparse
import itertools
import sys
import time

import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn

import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.lsc import MAG240MEvaluator
from ogb.nodeproppred import Evaluator
from tqdm import tqdm


def create_dataloader(
    name,
    graph,
    features,
    item_set,
    device,
    batch_size,
    fanouts,
    shuffle,
    num_workers,
):
    """Create a GraphBolt dataloader for training, validation or testing."""

    ###########################################################################
    # Initialize the ItemSampler to sample mini-batches from the dataset.
    # `item_set`:
    #   The set of items to sample from. This is typically the
    #   training, validation or test set.
    # `batch_size`:
    #   The number of nodes to sample in each mini-batch.
    # `shuffle`:
    #   Whether to shuffle the items in the dataset before sampling.
    datapipe = gb.ItemSampler(item_set, batch_size=batch_size, shuffle=shuffle)

    # Move the mini-batch to the appropriate device.
    # `device`:
    #   The device to move the mini-batch to.
    datapipe = datapipe.copy_to(device)

    # Sample neighbors for each seed node in the mini-batch.
    # `graph`:
    #   The graph(FusedCSCSamplingGraph) from which to sample neighbors.
    # `fanouts`:
    #   The number of neighbors to sample for each node in each layer.
    datapipe = datapipe.sample_neighbor(
        graph,
        fanouts=fanouts,
        overlap_fetch=args.overlap_graph_fetch,
        asynchronous=args.asynchronous,
    )

    # Fetch the features for each node in the mini-batch.
    # `features`:
    #   The feature store from which to fetch the features.
    # `node_feature_keys`:
    #   The node features to fetch. This is a dictionary where the keys are
    #   node types and the values are lists of feature names.
    node_feature_keys = {"paper": ["feat"]}
    if name == "ogb-lsc-mag240m":
        node_feature_keys["author"] = ["feat"]
        node_feature_keys["institution"] = ["feat"]
    datapipe = datapipe.fetch_feature(features, node_feature_keys)

    # Create a DataLoader from the datapipe.
    # `num_workers`:
    #   The number of worker processes to use for data loading.
    return gb.DataLoader(datapipe, num_workers=num_workers)


def extract_node_features(block, data, device):
    """Extract the node features from embedding layer or raw features."""
    node_features = {
        ntype: data.node_features[(ntype, "feat")]
        for ntype in block.srctypes
    }
    # Original feature data are stored in float16 while model weights are
    # float32, so we need to convert the features to float32.
    node_features = {
        k: v.to(device).float() for k, v in node_features.items()
    }
    return node_features


class RelGraphConvLayer(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        ntypes,
        relation_names,
        activation=None,
        dropout=0.0,
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.ntypes = ntypes
        self.relation_names = relation_names
        self.activation = activation

        ########################################################################
        # (HIGHLIGHT) HeteroGraphConv is a graph convolution operator over
        # heterogeneous graphs. A dictionary is passed where the key is the
        # relation name and the value is the instance of GraphConv. norm="right"
        # is to divide the aggregated messages by each node’s in-degrees, which
        # is equivalent to averaging the received messages. weight=False and
        # bias=False as we will use our own weight matrices defined later.
        ########################################################################
        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_size, out_size, norm="right", weight=False, bias=False
                )
                for rel in relation_names
            }
        )

        # Create a separate Linear layer for each relationship. Each
        # relationship has its own weights which will be applied to the node
        # features before performing convolution.
        self.weight = nn.ModuleDict(
            {
                rel_name: nn.Linear(in_size, out_size, bias=False)
                for rel_name in self.relation_names
            }
        )

        # Create a separate Linear layer for each node type.
        # loop_weights are used to update the output embedding of each target node
        # based on its own features, thereby allowing the model to refine the node
        # representations. Note that this does not imply the existence of self-loop
        # edges in the graph. It is similar to residual connection.
        self.loop_weights = nn.ModuleDict(
            {
                ntype: nn.Linear(in_size, out_size, bias=True)
                for ntype in self.ntypes
            }
        )

        self.loop_weights = nn.ModuleDict(
            {
                ntype: nn.Linear(in_size, out_size, bias=True)
                for ntype in self.ntypes
            }
        )

        self.dropout = nn.Dropout(dropout)
        # Initialize parameters of the model.
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.weight.values():
            layer.reset_parameters()

        for layer in self.loop_weights.values():
            layer.reset_parameters()

    def forward(self, g, inputs):
        """
        Parameters
        ----------
        g : DGLGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        # Create a deep copy of the graph g with features saved in local
        # frames to prevent side effects from modifying the graph.
        g = g.local_var()

        # Create a dictionary of weights for each relationship. The weights
        # are retrieved from the Linear layers defined earlier.
        weight_dict = {
            rel_name: {"weight": self.weight[rel_name].weight.T}
            for rel_name in self.relation_names
        }

        # Create a dictionary of node features for the destination nodes in
        # the graph. We slice the node features according to the number of
        # destination nodes of each type. This is necessary because when
        # incorporating the effect of self-loop edges, we perform computations
        # only on the destination nodes' features. By doing so, we ensure the
        # feature dimensions match and prevent any misuse of incorrect node
        # features.
        inputs_dst = {
            k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
        }
        # Apply the convolution operation on the graph. mod_kwargs are
        # additional arguments for each relation function defined in the
        # HeteroGraphConv. In this case, it's the weights for each relation.
        hs = self.conv(g, inputs, mod_kwargs=weight_dict)

        def _apply(ntype, h):
            # Apply the `loop_weight` to the input node features, effectively
            # acting as a residual connection. This allows the model to refine
            # node embeddings based on its current features.
            h = h + self.loop_weights[ntype](inputs_dst[ntype])
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        # Apply the function defined above for each node type. This will update
        # the node features using the `loop_weights`, apply the activation
        # function and dropout.
        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class EntityClassify(nn.Module):
    def __init__(self, graph, in_size, out_size):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.hidden_size = 64
        self.out_size = out_size

        # Generate and sort a list of unique edge types from the input graph.
        # eg. ['writes', 'cites']
        etypes = list(graph.edge_type_to_id.keys())
        etypes = [gb.etype_str_to_tuple(etype)[1] for etype in etypes]
        self.relation_names = etypes
        self.relation_names.sort()
        self.dropout = 0.5
        ntypes = list(graph.node_type_to_id.keys())
        self.layers = nn.ModuleList()

        # First layer: transform input features to hidden features. Use ReLU
        # as the activation function and apply dropout for regularization.
        self.layers.append(
            RelGraphConvLayer(
                self.in_size,
                self.hidden_size,
                ntypes,
                self.relation_names,
                activation=F.relu,
                dropout=self.dropout,
            )
        )

        # Second layer: transform hidden features to output features. No
        # activation function is applied at this stage.
        self.layers.append(
            RelGraphConvLayer(
                self.hidden_size,
                self.out_size,
                ntypes,
                self.relation_names,
                activation=None,
            )
        )

    def reset_parameters(self):
        # Reset the parameters of each layer.
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, blocks, h):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


@torch.no_grad()
def evaluate(
    name,
    g,
    model,
    device,
    item_set,
    features,
    num_workers,
):
    # Switches the model to evaluation mode.
    model.eval()
    category = "paper"
    # An evaluator for the dataset.
    if name == "ogbn-mag":
        evaluator = Evaluator(name=name)
    else:
        evaluator = MAG240MEvaluator()

    num_etype = len(g.num_edges)
    data_loader = create_dataloader(
        name,
        g,
        features,
        item_set,
        device,
        batch_size=4096,
        fanouts=[torch.full((num_etype,), 25), torch.full((num_etype,), 10)],
        shuffle=False,
        num_workers=num_workers,
    )

    # To store the predictions.
    y_hats = list()
    y_true = list()

    for data in tqdm(data_loader, desc="Inference"):
        # Convert MiniBatch to DGL Blocks and move them to the target device.
        blocks = [block.to(device) for block in data.blocks]
        node_features = extract_node_features(blocks[0], data, device)

        # Generate predictions.
        logits = model(blocks, node_features)

        logits = logits[category]

        # Apply softmax to the logits and get the prediction by selecting the
        # argmax.
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())
        y_true.append(data.labels[category].long())

    y_pred = torch.cat(y_hats, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_true = torch.unsqueeze(y_true, 1)

    if name == "ogb-lsc-mag240m":
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"]


def train(
    name,
    g,
    model,
    optimizer,
    train_set,
    valid_set,
    device,
    features,
    num_workers,
    num_epochs,
):
    print("Start to train...")
    category = "paper"

    num_etype = len(g.num_edges)
    data_loader = create_dataloader(
        name,
        g,
        features,
        train_set,
        device,
        batch_size=1024,
        fanouts=[torch.full((num_etype,), 25), torch.full((num_etype,), 10)],
        shuffle=True,
        num_workers=num_workers,
    )

    # Typically, the best Validation performance is obtained after
    # the 1st or 2nd epoch. This is why the max epoch is set to 3.
    for epoch in range(num_epochs):
        num_train = len(train_set)
        t0 = time.time()
        model.train()

        total_loss = 0

        for data in tqdm(data_loader, desc=f"Training~Epoch {epoch + 1:02d}"):
            # Convert MiniBatch to DGL Blocks and move them to the target
            # device.
            blocks = [block.to(device) for block in data.blocks]

            # Fetch the number of seed nodes in the batch.
            num_seeds = blocks[-1].num_dst_nodes(category)

            # Extract the node features from embedding layer or raw features.
            node_features = extract_node_features(blocks[0], data, device)

            # Reset gradients.
            optimizer.zero_grad()
            # Generate predictions.
            logits = model(blocks, node_features)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, data.labels[category].long())
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_seeds

        t1 = time.time()
        loss = total_loss / num_train

        # Evaluate the model on the val/test set.

        print("Evaluating the model on the validation set.")
        valid_acc = evaluate(
            name, g, model, node_embed, device, valid_set, features, num_workers
        )
        print("Finish evaluating on validation set.")

        print(
            f"Epoch: {epoch + 1:02d}, "
            f"Loss: {loss:.4f}, "
            f"Valid accuracy: {100 * valid_acc:.2f}%, "
            f"Time {t1 - t0:.4f}"
        )


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset.
    dataset = gb.BuiltinDataset(args.dataset).load()
    print(f"Loaded dataset: {dataset.tasks[0].metadata['name']}")

    g = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    test_set = dataset.tasks[0].test_set
    num_classes = dataset.tasks[0].metadata["num_classes"]

    # Move the dataset to the pinned memory to enable GPU access.
    args.overlap_graph_fetch = False
    args.asynchronous = False
    if device == torch.device("cuda"):
        g = g.pin_memory_()
        features = features.pin_memory_()
        # Enable optimizations for sampling on the GPU.
        args.overlap_graph_fetch = True
        args.asynchronous = True

    feat_size = features.size("node", "paper", "feat")[0]

    # Initialize the entity classification model.
    model = EntityClassify(g, feat_size, num_classes).to(device)

    print(
        "Number of model parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    train(
        args.dataset,
        g,
        model,
        optimizer,
        train_set,
        valid_set,
        device,
        features,
        args.num_workers,
        args.num_epochs,
    )

    print("Testing...")
    test_acc = evaluate(
        args.dataset,
        g,
        model,
        device,
        test_set,
        features,
        args.num_workers,
    )
    print(f"Test accuracy {test_acc*100:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphBolt RGCN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mag",
        choices=["ogbn-mag", "ogb-lsc-mag240m"],
        help="Dataset name. Possible values: ogbn-mag, ogb-lsc-mag240m",
    )
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)

    args = parser.parse_args()

    main(args)
