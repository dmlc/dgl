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

import dgl
import dgl.nn as dglnn

import psutil

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddReverse, Compose, ToSimple
from dgl.nn import HeteroEmbedding
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
from tqdm import tqdm


def prepare_data(args, device):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    # Get train/valid/test index.
    split_idx = dataset.get_idx_split()

    # - graph: dgl graph object.
    # - label: torch tensor of shape (num_nodes, num_tasks).
    g, labels = dataset[0]

    # Flatten the labels for "paper" type nodes. This step reduces the
    # dimensionality of the labels. We need to flatten the labels because
    # the model requires a 1-dimensional label tensor.
    labels = labels["paper"].flatten()

    # Apply transformation to the graph.
    # - "ToSimple()" removes multi-edge between two nodes.
    # - "AddReverse()" adds reverse edges to the graph.
    transform = Compose([ToSimple(), AddReverse()])
    g = transform(g)

    print(f"Loaded graph: {g}")

    # Initialize a train sampler that samples neighbors for multi-layer graph
    # convolution. It samples 25 and 20 neighbors for the first and second
    # layers respectively.
    sampler = dgl.dataloading.MultiLayerNeighborSampler([25, 20])
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

    return g, labels, dataset.num_classes, split_idx, train_loader


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
    def __init__(self, g, in_size, out_dim):
        super(EntityClassify, self).__init__()
        self.in_size = in_size
        self.hidden_size = 64
        self.out_dim = out_dim

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
                self.out_dim,
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


class Logger(object):
    r"""
    This class was taken directly from the PyG implementation and can be found
    here: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppre
    d/mag/logger.py

    This was done to ensure that performance was measured in precisely the same
    way
    """

    def __init__(self, runs):
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * th.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * th.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = th.tensor(best_results)

            print("All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


def train(
    g,
    model,
    node_embed,
    optimizer,
    train_loader,
    split_idx,
    labels,
    logger,
    device,
    run,
):
    print("start training...")
    category = "paper"

    # Typically, the best Validation performance is obtained after
    # the 1st or 2nd epoch. This is why the max epoch is set to 3.
    for epoch in range(3):
        num_train = split_idx["train"][category].shape[0]
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
            input_nodes_indexes = input_nodes[category].to(g.device)
            seeds = seeds.to(labels.device)

            # Extract node embeddings for the input nodes.
            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features. Corresponds to the content
            # in the function `rel_graph_embed` comment.
            emb.update(
                {category: g.ndata["feat"][category][input_nodes_indexes]}
            )

            emb = {k: e.to(device) for k, e in emb.items()}
            lbl = labels[seeds].to(device)

            # Reset gradients.
            optimizer.zero_grad()
            # Generate predictions.
            logits = model(emb, blocks)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size

        loss = total_loss / num_train

        # Evaluate the model on the test set.
        result = test(g, model, node_embed, labels, device, split_idx)
        logger.add_result(run, result)
        train_acc, valid_acc, test_acc = result
        print(
            f"Run: {run + 1:02d}, "
            f"Epoch: {epoch +1 :02d}, "
            f"Loss: {loss:.4f}, "
            f"Train: {100 * train_acc:.2f}%, "
            f"Valid: {100 * valid_acc:.2f}%, "
            f"Test: {100 * test_acc:.2f}%"
        )

    return logger


@th.no_grad()
def test(g, model, node_embed, y_true, device, split_idx):
    # Switches the model to evaluation mode.
    model.eval()
    category = "paper"
    # An evaluator for the dataset 'ogbn-mag'.
    evaluator = Evaluator(name="ogbn-mag")

    # Initialize a neighbor sampler that samples all neighbors. The model
    # has 2 GNN layers, so we create a sampler of 2 layers.
    ######################################################################
    # [Why we need to sample all neighbors?]
    # During the testing phase, we use a `MultiLayerFullNeighborSampler` to
    # sample all neighbors for each node. This is done to achieve the most
    # accurate evaluation of the model's performance, despite the increased
    # computational cost. This contrasts with the training phase where we
    # prefer a balance between computational efficiency and model accuracy,
    # hence only a subset of neighbors is sampled.
    ######################################################################
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    loader = dgl.dataloading.DataLoader(
        g,
        {category: th.arange(g.num_nodes(category))},
        sampler,
        batch_size=16384,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    # To store the predictions.
    y_hats = list()

    for input_nodes, seeds, blocks in tqdm(loader, desc="Inference"):
        blocks = [blk.to(device) for blk in blocks]
        # We only predict the nodes with type "category".
        seeds = seeds[category]
        input_nodes_indexes = input_nodes[category].to(g.device)

        # Extract node embeddings for the input nodes.
        emb = extract_embed(node_embed, input_nodes)
        # Add the batch's raw "paper" features.
        # Corresponds to the content in the function `rel_graph_embed` comment.
        emb.update({category: g.ndata["feat"][category][input_nodes_indexes]})
        emb = {k: e.to(device) for k, e in emb.items()}

        # Generate predictions.
        logits = model(emb, blocks)[category]
        # Apply softmax to the logits and get the prediction by selecting the
        # argmax.
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())

    y_pred = th.cat(y_hats, dim=0)
    y_true = th.unsqueeze(y_true, 1)

    # Calculate the accuracy of the predictions for the train, valid and
    # test splits.
    train_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["train"]["paper"]],
            "y_pred": y_pred[split_idx["train"]["paper"]],
        }
    )["acc"]
    valid_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["valid"]["paper"]],
            "y_pred": y_pred[split_idx["valid"]["paper"]],
        }
    )["acc"]
    test_acc = evaluator.eval(
        {
            "y_true": y_true[split_idx["test"]["paper"]],
            "y_pred": y_pred[split_idx["test"]["paper"]],
        }
    )["acc"]

    return train_acc, valid_acc, test_acc


def main(args):
    device = "cuda:0" if th.cuda.is_available() else "cpu"

    # Initialize a logger.
    logger = Logger(args.runs)

    # Prepare the data.
    g, labels, num_classes, split_idx, train_loader = prepare_data(args, device)

    # Create the embedding layer and move it to the appropriate device.
    embed_layer = rel_graph_embed(g, 128).to(device)

    # Initialize the entity classification model.
    model = EntityClassify(g, 128, num_classes).to(device)

    print(
        "Number of embedding parameters: "
        f"{sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(
        "Number of model parameters: "
        f"{sum(p.numel() for p in model.parameters())}"
    )

    for run in range(args.runs):
        try:
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
            model.parameters(), embed_layer.parameters()
        )
        optimizer = th.optim.Adam(all_params, lr=0.01)

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
        logger = train(
            g,
            model,
            embed_layer,
            optimizer,
            train_loader,
            split_idx,
            labels,
            logger,
            device,
            run,
        )
        logger.print_statistics(run)

    print("Final performance: ")
    logger.print_statistics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RGCN")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    main(args)
