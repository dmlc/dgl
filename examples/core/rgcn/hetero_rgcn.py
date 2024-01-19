"""
This script, `hetero_rgcn.py`, trains and tests a Relational Graph
Convolutional Network (R-GCN) model for node classification on the
Open Graph Benchmark (OGB) dataset "ogbn-mag". For more details on
"ogbn-mag", please refer to the OGB website:
(https://ogb.stanford.edu/docs/linkprop/)

Paper [Modeling Relational Data with Graph Convolutional Networks]
(https://arxiv.org/abs/1703.06103).

Generation of graph embeddings is the main difference between homograph
node classification and heterograph node classification:
- Homograph: Since all nodes and edges are of the same type, embeddings
  can be generated using a unified approach. Type-specific handling is
  typically not required.
- Heterograph: Due to the existence of multiple types of nodes and edges,
  specific embeddings need to be generated for each type. This allows for
  a more nuanced capture of the complex structure and semantic information
  within the heterograph.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> prepare_data
│     │
│     └───> Load and preprocess dataset
│
├───> rel_graph_embed [HIGHLIGHT]
│     │
│     └───> Generate graph embeddings
│
├───> Instantiate RGCN model
│     │
│     ├───> RelGraphConvLayer (input to hidden)
│     │
│     └───> RelGraphConvLayer (hidden to output)
│
└───> train
      │
      │
      └───> Training loop
            │
            ├───> EntityClassify.forward (RGCN model forward pass)
            │
            └───> test
                  │
                  └───> EntityClassify.evaluate
"""

import argparse
import itertools
import sys
import time

import dgl
import dgl.nn as dglnn
import numpy as np

import psutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddReverse, Compose, ToSimple
from dgl.nn import HeteroEmbedding
from ogb.lsc import MAG240MDataset, MAG240MEvaluator
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm


def prepare_data(args, device):
    feats = {}
    if args.dataset == "ogbn-mag":
        dataset = DglNodePropPredDataset(name="ogbn-mag", root=args.rootdir)

        # - graph: dgl graph object.
        # - label: torch tensor of shape (num_nodes, num_tasks).
        g, labels = dataset[0]

        # Flatten the labels for "paper" type nodes. This step reduces the
        # dimensionality of the labels. We need to flatten the labels because
        # the model requires a 1-dimensional label tensor.
        labels = labels["paper"].flatten().long()

        # Apply transformation to the graph.
        # - "ToSimple()" removes multi-edge between two nodes.
        # - "AddReverse()" adds reverse edges to the graph.
        print("Start to transform graph. This may take a while...")
        transform = Compose([ToSimple(), AddReverse()])
        g = transform(g)
    else:
        dataset = MAG240MDataset(root=args.rootdir)
        (g,), _ = dgl.load_graphs(args.graph_path)
        g = g.formats(["csc"])
        labels = torch.as_tensor(dataset.paper_label).long()
        # As feature data is too large to fit in memory, we read it from disk.
        feats["paper"] = torch.as_tensor(
            np.load(args.paper_feature_path, mmap_mode="r+")
        )
        feats["author"] = torch.as_tensor(
            np.load(args.author_feature_path, mmap_mode="r+")
        )
        feats["institution"] = torch.as_tensor(
            np.load(args.inst_feature_path, mmap_mode="r+")
        )
    print(f"Loaded graph: {g}")

    # Get train/valid/test index.
    split_idx = dataset.get_idx_split()
    if args.dataset == "ogb-lsc-mag240m":
        split_idx = {
            split_type: {"paper": split_idx[split_type]}
            for split_type in split_idx
        }

    # Initialize a train sampler that samples neighbors for multi-layer graph
    # convolution. It samples 25 and 10 neighbors for the first and second
    # layers respectively.
    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 10], fused=False)
    num_workers = args.num_workers
    train_loader = dgl.dataloading.DataLoader(
        g,
        split_idx["train"],
        sampler,
        batch_size=1024,
        shuffle=True,
        num_workers=num_workers,
        device=device,
    )

    return g, labels, dataset.num_classes, split_idx, train_loader, feats


def extract_embed(node_embed, input_nodes):
    emb = node_embed(
        {ntype: input_nodes[ntype] for ntype in input_nodes if ntype != "paper"}
    )
    return emb


def rel_graph_embed(graph, embed_size):
    """Initialize a heterogenous embedding layer for all node types in the
    graph, except for the "paper" node type.

    The function constructs a dictionary 'node_num', where the keys are node
    types (ntype) and the values are the number of nodes for each type. This
    dictionary is used to create a HeteroEmbedding instance.

    (HIGHLIGHT)
    A HeteroEmbedding instance holds separate embedding layers for each node
    type, each with its own feature space of dimensionality
    (node_num[ntype], embed_size), where 'node_num[ntype]' is the number of
    nodes of type 'ntype' and 'embed_size' is the embedding dimension.

    The "paper" node type is specifically excluded, possibly because these nodes
    might already have predefined feature representations, and therefore, do not
    require an additional embedding layer.

    Parameters
    ----------
    graph : DGLGraph
        The graph for which to create the heterogenous embedding layer.
    embed_size : int
        The size of the embedding vectors.

    Returns
    --------
    HeteroEmbedding
        A heterogenous embedding layer for all node types in the graph, except
        for the "paper" node type.
    """
    node_num = {}
    for ntype in graph.ntypes:
        # Skip the "paper" node type.
        if ntype == "paper":
            continue
        node_num[ntype] = graph.num_nodes(ntype)
    return HeteroEmbedding(node_num, embed_size)


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
    def __init__(self, g, in_size, out_size):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.hidden_size = 64
        self.out_size = out_size

        # Generate and sort a list of unique edge types from the input graph.
        # eg. ['writes', 'cites']
        self.relation_names = list(set(g.etypes))
        self.relation_names.sort()
        self.dropout = 0.5

        self.layers = nn.ModuleList()

        # First layer: transform input features to hidden features. Use ReLU
        # as the activation function and apply dropout for regularization.
        self.layers.append(
            RelGraphConvLayer(
                self.in_size,
                self.hidden_size,
                g.ntypes,
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
                g.ntypes,
                self.relation_names,
                activation=None,
            )
        )

    def reset_parameters(self):
        # Reset the parameters of each layer.
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


def extract_node_features(name, g, input_nodes, node_embed, feats, device):
    """Extract the node features from embedding layer or raw features."""
    if name == "ogbn-mag":
        # Extract node embeddings for the input nodes.
        node_features = extract_embed(node_embed, input_nodes)
        # Add the batch's raw "paper" features. Corresponds to the content
        # in the function `rel_graph_embed` comment.
        node_features.update(
            {"paper": g.ndata["feat"]["paper"][input_nodes["paper"].cpu()]}
        )
        node_features = {k: e.to(device) for k, e in node_features.items()}
    else:
        node_features = {
            ntype: feats[ntype][input_nodes[ntype].cpu()].to(device)
            for ntype in input_nodes
        }
        # Original feature data are stored in float16 while model weights are
        # float32, so we need to convert the features to float32.
        # [TODO] Enable mixed precision training on GPU.
        node_features = {k: v.float() for k, v in node_features.items()}
    return node_features


def train(
    dataset,
    g,
    feats,
    model,
    node_embed,
    optimizer,
    train_loader,
    split_idx,
    labels,
    device,
):
    print("Start training...")
    category = "paper"

    # Typically, the best Validation performance is obtained after
    # the 1st or 2nd epoch. This is why the max epoch is set to 3.
    for epoch in range(3):
        num_train = split_idx["train"][category].shape[0]
        t0 = time.time()
        model.train()

        total_loss = 0

        for input_nodes, seeds, blocks in tqdm(
            train_loader, desc=f"Epoch {epoch:02d}"
        ):
            # Move the input data onto the device.
            blocks = [blk.to(device) for blk in blocks]
            # We only predict the nodes with type "category".
            seeds = seeds[category]
            batch_size = seeds.shape[0]

            # Extract the node features from embedding layer or raw features.
            node_features = extract_node_features(
                dataset, g, input_nodes, node_embed, feats, device
            )
            lbl = labels[seeds.cpu()].to(device)

            # Reset gradients.
            optimizer.zero_grad()
            # Generate predictions.
            logits = model(node_features, blocks)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        t1 = time.time()
        loss = total_loss / num_train

        # Evaluate the model on the val/test set.
        valid_acc = evaluate(
            dataset,
            g,
            feats,
            model,
            node_embed,
            labels,
            device,
            split_idx["valid"],
        )
        test_key = "test" if dataset == "ogbn-mag" else "test-dev"
        test_acc = evaluate(
            dataset,
            g,
            feats,
            model,
            node_embed,
            labels,
            device,
            split_idx[test_key],
            save_test_submission=(dataset == "ogb-lsc-mag240m"),
        )
        print(
            f"Epoch: {epoch +1 :02d}, "
            f"Loss: {loss:.4f}, "
            f"Valid: {100 * valid_acc:.2f}%, "
            f"Test: {100 * test_acc:.2f}%, "
            f"Time {t1 - t0:.4f}"
        )


@torch.no_grad()
def evaluate(
    dataset,
    g,
    feats,
    model,
    node_embed,
    labels,
    device,
    idx,
    save_test_submission=False,
):
    # Switches the model to evaluation mode.
    model.eval()
    category = "paper"
    if dataset == "ogbn-mag":
        evaluator = Evaluator(name="ogbn-mag")
    else:
        evaluator = MAG240MEvaluator()

    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 10], fused=False)
    dataloader = dgl.dataloading.DataLoader(
        g,
        idx,
        sampler,
        batch_size=4096,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    # To store the predictions.
    y_hats = list()
    y_true = list()

    for input_nodes, seeds, blocks in tqdm(dataloader, desc="Inference"):
        blocks = [blk.to(device) for blk in blocks]
        # We only predict the nodes with type "category".
        node_features = extract_node_features(
            dataset, g, input_nodes, node_embed, feats, device
        )

        # Generate predictions.
        logits = model(node_features, blocks)[category]
        # Apply softmax to the logits and get the prediction by selecting the
        # argmax.
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())
        y_true.append(labels[seeds["paper"].cpu()])

    y_pred = torch.cat(y_hats, dim=0)
    y_true = torch.cat(y_true, dim=0)
    y_true = torch.unsqueeze(y_true, 1)

    if dataset == "ogb-lsc-mag240m":
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

    if save_test_submission:
        evaluator.save_test_submission(
            input_dict={"y_pred": y_pred}, dir_path=".", mode="test-dev"
        )
    return evaluator.eval({"y_true": y_true, "y_pred": y_pred})["acc"]


def main(args):
    device = (
        "cuda:0" if torch.cuda.is_available() and args.num_gpus > 0 else "cpu"
    )

    # Prepare the data.
    g, labels, num_classes, split_idx, train_loader, feats = prepare_data(
        args, device
    )

    feat_size = 128 if args.dataset == "ogbn-mag" else 768

    # Create the embedding layer and move it to the appropriate device.
    embed_layer = None
    if args.dataset == "ogbn-mag":
        embed_layer = rel_graph_embed(g, feat_size).to(device)
        print(
            "Number of embedding parameters: "
            f"{sum(p.numel() for p in embed_layer.parameters())}"
        )

    # Initialize the entity classification model.
    model = EntityClassify(g, feat_size, num_classes).to(device)

    print(
        "Number of model parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    try:
        if embed_layer is not None:
            embed_layer.reset_parameters()
        model.reset_parameters()
    except:
        # Old pytorch version doesn't support reset_parameters() API.
        ##################################################################
        # [Why we need to reset the parameters?]
        # If parameters are not reset, the model will start with the
        # parameters learned from the last run, potentially resulting
        # in biased outcomes or sub-optimal performance if the model was
        # previously stuck in a poor local minimum.
        ##################################################################
        pass

    # `itertools.chain()` is a function in Python's itertools module.
    # It is used to flatten a list of iterables, making them act as
    # one big iterable.
    # In this context, the following code is used to create a single
    # iterable over the parameters of both the model and the embed_layer,
    # which is passed to the optimizer. The optimizer then updates all
    # these parameters during the training process.
    all_params = itertools.chain(
        model.parameters(),
        [] if embed_layer is None else embed_layer.parameters(),
    )
    optimizer = torch.optim.Adam(all_params, lr=0.01)

    # `expected_max`` is the number of physical cores on your machine.
    # The `logical` parameter, when set to False, ensures that the count
    # returned is the number of physical cores instead of logical cores
    # (which could be higher due to technologies like Hyper-Threading).
    expected_max = int(psutil.cpu_count(logical=False))
    if args.num_workers >= expected_max:
        print(
            "[ERROR] You specified num_workers are larger than physical"
            f"cores, please set any number less than {expected_max}",
            file=sys.stderr,
        )
    train(
        args.dataset,
        g,
        feats,
        model,
        embed_layer,
        optimizer,
        train_loader,
        split_idx,
        labels,
        device,
    )

    print("Testing...")
    test_key = "test" if args.dataset == "ogbn-mag" else "test-dev"
    test_acc = evaluate(
        args.dataset,
        g,
        feats,
        model,
        embed_layer,
        labels,
        device,
        split_idx[test_key],
        save_test_submission=(args.dataset == "ogb-lsc-mag240m"),
    )
    print(f"Test accuracy {test_acc*100:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbn-mag",
        help="Dataset for train: ogbn-mag, ogb-lsc-mag240m",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=0,
        help="Number of GPUs. Use 0 for CPU training.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading.",
    )
    parser.add_argument(
        "--rootdir",
        type=str,
        default="./dataset/",
        help="Directory to download the OGB dataset.",
    )
    parser.add_argument(
        "--graph_path",
        type=str,
        default="./graph.dgl",
        help="Path to the graph file.",
    )
    parser.add_argument(
        "--paper_feature_path",
        type=str,
        default="./paper-feat.npy",
        help="Path to the features of paper nodes.",
    )
    parser.add_argument(
        "--author_feature_path",
        type=str,
        default="./author-feat.npy",
        help="Path to the features of author nodes.",
    )
    parser.add_argument(
        "--inst_feature_path",
        type=str,
        default="./inst-feat.npy",
        help="Path to the features of institution nodes.",
    )

    args = parser.parse_args()

    main(args)
