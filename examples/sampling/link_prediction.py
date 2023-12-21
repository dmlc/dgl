"""
This script trains and tests a GraphSAGE model for link prediction on
large graphs using efficient and tailor-made neighbor sampling.

Paper: [Inductive Representation Learning on Large Graphs]
(https://arxiv.org/abs/1706.02216)

While node classification predicts labels for nodes based on their
local neighborhoods, link prediction assesses the likelihood of an edge
existing between two nodes, necessitating different sampling strategies
that account for pairs of nodes and their joint neighborhoods.

Before reading this example, please familiar yourself with graphsage node
classification by reading the example in the
`examples/core/graphsage/node_classification.py`

If you want to train graphsage on a large graph in a distributed fashion, read
the example in the `examples/distributed/graphsage/`.

This flowchart describes the main functional sequence of the provided example.
main
│
├───> Load and preprocess dataset
│
├───> Instantiate SAGE model
│
├───> train
│     │
│     ├───> NeighborSampler (HIGHLIGHT)
│     │
│     └───> Training loop
│           │
│           └───> SAGE.forward
│
└───> evaluate
      │
      └───> SAGE.inference
            │
            └───> MultiLayerFullNeighborSampler (HIGHLIGHT)
"""

import argparse
import time

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator


def to_bidirected_with_reverse_mapping(g):
    """Convert the graph to bidirectional and return the reverse mapping.

    This function transforms the input graph into its bidirectional form. It
    then returns the newly formed bidirectional graph and the mapping that
    represents the reverse edges. The function does not work with graphs that
    have self-loops.

    Parameters:
    ----------
    g : DGLGraph
        Input graph.

    Returns:
    -------
    DGLGraph :
        Bidirectional graph.
    Tensor :
        Mapping to reverse edges.
    """
    # First, add reverse edges to the graph, effectively making it
    # bidirectional. Then, simplify the resulting graph by merging any duplicate
    # edges. The resulting simplified graph is stored in `g_simple`, and
    # `mapping` provides information on how edges in `g_simple` correspond to
    # edges in the original graph.
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )

    # The `return_counts` option in `dgl.to_simple` returns the count of how
    # many times each edge in the simplified graph corresponds to an edge in the
    # original graph. This count is saved in the edge data of the returned
    # graph with the key "count".
    c = g_simple.edata["count"]
    num_edges = g.num_edges()

    # `mapping_offset` is an auxiliary tensor used to understand how edges in
    # the simplified bidirectional graph (g_simple) relate to the edges in the
    # original graph.
    mapping_offset = torch.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )

    # Calculate the cumulative sum of counts to determine boundaries for each
    # unique edge.
    mapping_offset[1:] = c.cumsum(0)

    # Sort the mapping tensor to group the same edge indices.
    idx = mapping.argsort()

    # Using the previously computed `mapping_offset`, it extracts the first
    # index of each group, which represents the unique edge indices from the
    # sorted mapping.
    idx_uniq = idx[mapping_offset[:-1]]

    # If an edge index is greater than or equal to the number of edges in the
    # original graph, it indicates that this edge is a reversed edge, and the
    # original edge index for it is (idx_uniq - num_edges). Otherwise, its
    # reverse edge index is (idx_uniq + num_edges).
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]

    # Sanity check to ensure valid mapping.
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # Three-layer GraphSAGE-mean.
        self.layers.append(dglnn.SAGEConv(in_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hidden_size, hidden_size, "mean"))
        self.hidden_size = hidden_size
        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        hidden_pos = self.predictor(hidden_x[pos_src] * hidden_x[pos_dst])
        hidden_neg = self.predictor(hidden_x[neg_src] * hidden_x[neg_dst])
        return hidden_pos, hidden_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
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
        # feature named 'feat' will be pre-fetched.
        #
        # `prefetch` in DGL initiates data fetching operations in parallel
        # with model computations. This ensures data is ready when the
        # computation needs it, thereby eliminating waiting times between
        # fetching and computing steps and reducing the I/O overhead during
        # the training process.
        #
        # The difference between whether to use prefetch or not is shown:
        #
        # Without Prefetch:
        # Fetch1 ──> Compute1 ──> Fetch2 ──> Compute2 ──> Fetch3 ──> Compute3
        #
        # With Prefetch:
        # Fetch1 ──> Fetch2 ──> Fetch3
        #    │          │          │
        #    └─Compute1 └─Compute2 └─Compute3
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
        # Enable pin_memory for faster CPU to GPU data transfer if the model is
        # running on a GPU.
        pin_memory = buffer_device != device
        for layer_idx, layer in enumerate(self.layers):
            is_last_layer = layer_idx == len(self.layers) - 1
            y = torch.empty(
                g.num_nodes(),
                self.hidden_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="Inference"
            ):
                x = feat[input_nodes]
                hidden_x = layer(blocks[0], x)
                if not is_last_layer:
                    hidden_x = F.relu(hidden_x)
                y[output_nodes] = hidden_x.to(buffer_device)
            feat = y
        return y


@torch.no_grad()
def compute_mrr(
    model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500
):
    """Compute the Mean Reciprocal Rank (MRR) for given source and destination
    nodes.

    This function computes the MRR for a set of node pairs, dividing the task
    into batches to handle potentially large graphs.
    """
    rr = torch.zeros(src.shape[0])
    # Loop over node pairs in batches.
    for start in tqdm.trange(0, src.shape[0], batch_size, desc="Evaluate"):
        end = min(start + batch_size, src.shape[0])

        # Concatenate positive and negative destination nodes.
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)

        # Fetch embeddings for current batch of source and destination nodes.
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)

        # Compute prediction scores using the model.
        pred = model.predictor(h_src * h_dst).squeeze(-1)

        # Evaluate the predictions to obtain MRR values.
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()


@torch.no_grad()
def evaluate(device, graph, edge_split, model, batch_size):
    """Evaluate the model on validation and test sets."""
    model.eval()
    evaluator = Evaluator(name="ogbl-citation2")

    # Compute node embeddings for the entire graph.
    node_emb = model.inference(graph, device, batch_size)
    results = []

    # Loop over both validation and test sets.
    for split in ["valid", "test"]:
        src = edge_split[split]["source_node"].to(node_emb.device)
        dst = edge_split[split]["target_node"].to(node_emb.device)
        neg_dst = edge_split[split]["target_node_neg"].to(node_emb.device)

        # Compute MRR values for the current split.
        results.append(
            compute_mrr(model, evaluator, node_emb, src, dst, neg_dst, device)
        )
    return results


def train(
    args, device, g, reverse_eids, seed_edges, model, use_uva, fused_sampling
):
    #####################################################################
    # (HIGHLIGHT) Instantiate a NeighborSampler object for efficient
    # training of Graph Neural Networks (GNNs) on large-scale graphs.
    #
    # The argument [15, 10, 5] sets the number of neighbors (fanout)
    # to be sampled at each layer. Here, we have three layers, and
    # 15/10/5 neighbors will be randomly selected for each node at each
    # layer.
    #
    # The 'prefetch_node_feats' parameter specify the node features that
    # needs to be pre-fetched during sampling. More details about
    # `prefetch` can be found in the `SAGE.inference` function.
    #
    # (HIGHLIGHT) Modify the NeighborSampler for Edge Prediction
    #
    # This `as_edge_prediction_sampler` augments the original NeighborSampler
    # to specifically handle edge prediction tasks, where not only the
    # structure but also the relationships between nodes (edges) are of
    # importance.
    #
    # - `exclude="reverse_id"` ensures that the edges corresponding to the
    #   reverse of the original edges are excluded during sampling, given that
    #   reverse edges can introduce unnecessary redundancy in edge prediction.
    #
    # - `reverse_eids=reverse_eids` specifies the IDs of the reverse edges.
    #   This information is vital so the sampler knows which edges to avoid.
    #
    # - The negative sampling strategy is specified using the
    #   `negative_sampler`. Here, a uniform negative sampling method is
    #   employed, where a negative sample (an edge that doesn't exist in the
    #   original graph) is uniformly drawn from the set of all possible edges.
    #
    # The modified sampler is tailor-made for scenarios where the goal is
    # not just to learn node representations, but also to predict the
    # likelihood of an edge existing between two nodes (link prediction).
    #####################################################################
    sampler = NeighborSampler(
        [15, 10, 5],
        prefetch_node_feats=["feat"],
        fused=fused_sampling,
    )
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id" if args.exclude_edges else None,
        reverse_eids=reverse_eids if args.exclude_edges else None,
        negative_sampler=negative_sampler.Uniform(1),
    )

    dataloader = DataLoader(
        g,
        seed_edges,
        sampler,
        device=device,
        batch_size=args.train_batch_size,
        shuffle=True,
        drop_last=False,
        # If `g` is on gpu or `use_uva` is True, `num_workers` must be zero,
        # otherwise it will cause error.
        num_workers=0,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_epoch_time = time.time()
        # A block is a graph consisting of two sets of nodes: the
        # source nodes and destination nodes. The source and destination
        # nodes can have multiple node types. All the edges connect from
        # source nodes to destination nodes.
        # For more details: https://discuss.dgl.ai/t/what-is-the-block/2932.
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader
        ):
            # The input features from the source nodes in the first layer's
            # computation graph.
            x = blocks[0].srcdata["feat"]
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])

            # Create true labels for positive and negative samples.
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])

            # Compute the binary cross-entropy loss.
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (it + 1) == args.early_stop:
                break
        end_epoch_time = time.time()
        print(
            f"Epoch {epoch:05d} | "
            f"Loss {total_loss / (it + 1):.4f} | "
            f"Time {(end_epoch_time - start_epoch_time):.4f} s"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate. Default: 0.0005",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=512,
        help="Batch size for training. Default: 512",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1024,
        help="Batch size during evaluation. Default: 1024",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="0 means no early stop, otherwise stop at the input-th step",
    )
    parser.add_argument(
        "--exclude-edges",
        type=int,
        default=1,
        help="Whether to exclude reverse edges during sampling. Default: 1",
    )
    parser.add_argument(
        "--compare-graphbolt",
        action="store_true",
        help="Compare with GraphBolt",
    )
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed "
        "training, 'puregpu' for pure-GPU training.",
    )
    return parser.parse_args()


def main(args):
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # Load and preprocess dataset.
    print("Loading data")
    dataset = DglLinkPropPredDataset("ogbl-citation2")
    g = dataset[0]
    if args.compare_graphbolt:
        fused_sampling = False
    else:
        fused_sampling = True
        g = g.to("cuda" if args.mode == "puregpu" else "cpu")

    # Whether use Unified Virtual Addressing (UVA) for CUDA computation.
    use_uva = args.mode == "mixed"
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # Convert the graph to its bidirectional form.
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(g.device)
    seed_edges = torch.arange(g.num_edges()).to(g.device)
    edge_split = dataset.get_edge_split()

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256).to(device)

    # Model training.
    print("Training...")
    train(
        args,
        device,
        g,
        reverse_eids,
        seed_edges,
        model,
        use_uva,
        fused_sampling,
    )

    # Validate/Test the model.
    print("Validation/Testing...")
    valid_mrr, test_mrr = evaluate(
        device, g, edge_split, model, batch_size=args.eval_batch_size
    )
    print(
        f"Validation MRR {valid_mrr.item():.4f}, Test MRR {test_mrr.item():.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
