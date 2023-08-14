import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator


def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping``
    where ``mapping[i]`` is the reverse edge of edge ID ``i``. Does not work
    with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g),
        return_counts="count",
        writeback_mapping=True,
    )

    # Construct the reverse edge mapping.
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]

    # Sanity check to ensure the bidirectionality.
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

    def inference(self, g):
        """Compute node embeddings for the entire graph.

        This function performs inference using the GraphSAGE layers
        and computes node embeddings for the entire graph.
        """
        # Retrieve the initial node features.
        feat = g.ndata["feat"]

        for layer_idx, layer in enumerate(self.layers):
            y = layer(g, feat)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                y = F.relu(y)
            feat = y
        return feat


def compute_mrr(
    model, evaluator, node_emb, src, dst, neg_dst, device, batch_size=500
):
    """Compute Mean Reciprocal Rank (MRR) in batches."""
    rr = torch.zeros(src.shape[0])
    for start in tqdm.trange(0, src.shape[0], batch_size, desc="Evaluate"):
        end = min(start + batch_size, src.shape[0])
        all_dst = torch.cat([dst[start:end, None], neg_dst[start:end]], 1)
        h_src = node_emb[src[start:end]][:, None, :].to(device)
        h_dst = node_emb[all_dst.view(-1)].view(*all_dst.shape, -1).to(device)
        pred = model.predictor(h_src * h_dst).squeeze(-1)
        input_dict = {"y_pred_pos": pred[:, 0], "y_pred_neg": pred[:, 1:]}
        rr[start:end] = evaluator.eval(input_dict)["mrr_list"]
    return rr.mean()


def evaluate(device, graph, edge_split, model):
    model.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    with torch.no_grad():
        node_emb = model.inference(graph)
        results = []
        for split in ["valid", "test"]:
            src = edge_split[split]["source_node"].to(node_emb.device)
            dst = edge_split[split]["target_node"].to(node_emb.device)
            neg_dst = edge_split[split]["target_node_neg"].to(node_emb.device)
            results.append(
                compute_mrr(
                    model, evaluator, node_emb, src, dst, neg_dst, device
                )
            )
    return results


def construct_negative_graph(graph, k):
    src, dst = graph.edges()

    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.num_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.num_nodes())


def train(device, g, model):
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)

    negative_g = construct_negative_graph(g, 1)
    for epoch in range(10):
        model.train()
        x = g.ndata["feat"].to(device)
        pos_score, neg_score = model(g, negative_g, [g, g, g], x)
        score = torch.cat([pos_score, neg_score])
        pos_label = torch.ones_like(pos_score).to(device)
        neg_label = torch.zeros_like(neg_score).to(device)
        labels = torch.cat([pos_label, neg_label])

        loss = F.binary_cross_entropy_with_logits(score, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"Epoch {epoch + 1:05d} | Loss {loss.item():.4f}")


def parse_args():
    """Parses the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="cpu",
        choices=["cpu", "gpu"],
        help="Training mode. 'cpu' for CPU training,"
        "'gpu' for pure-GPU training.",
    )
    return parser.parse_args()


def main(args):
    """Main function for the example."""
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # Load and preprocess dataset.
    print("Loading data...")
    dataset = DglLinkPropPredDataset("ogbl-citation2")
    g = dataset[0]

    g = g.to("cuda" if args.mode == "gpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    # TODO: Can we remove this in full-graph training?
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    edge_split = dataset.get_edge_split()

    # Create GraphSAGE model.
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256).to(device)

    # Model training.
    print("Training...")
    train(device, g, model)

    # Validate/test the model.
    print("Validation/Testing...")
    valid_mrr, test_mrr = evaluate(device, g, edge_split, model)
    print(
        f"Validation MRR {valid_mrr.item():.4f}, Test MRR {test_mrr.item():.4f}"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
