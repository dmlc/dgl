import argparse

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
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )
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
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.hid_size = hid_size
        self.predictor = nn.Sequential(
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
        pos_src, pos_dst = pair_graph.edges()
        neg_src, neg_dst = neg_pair_graph.edges()
        h_pos = self.predictor(h[pos_src] * h[pos_dst])
        h_neg = self.predictor(h[neg_src] * h[neg_dst])
        return h_pos, h_neg

    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
        feat = g.ndata["feat"]
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
        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="Inference"
            ):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y


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


def evaluate(device, graph, edge_split, model, batch_size):
    model.eval()
    evaluator = Evaluator(name="ogbl-citation2")
    with torch.no_grad():
        node_emb = model.inference(graph, device, batch_size)
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


def train(args, device, g, reverse_eids, seed_edges, model):
    # create sampler & dataloader
    sampler = NeighborSampler([15, 10, 5], prefetch_node_feats=["feat"])
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )
    use_uva = args.mode == "mixed"
    dataloader = DataLoader(
        g,
        seed_edges,
        sampler,
        device=device,
        batch_size=512,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    for epoch in range(10):
        model.train()
        total_loss = 0
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader
        ):
            x = blocks[0].srcdata["feat"]
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (it + 1) == 1000:
                break
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
    dataset = DglLinkPropPredDataset("ogbl-citation2")
    g = dataset[0]
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    edge_split = dataset.get_edge_split()

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = SAGE(in_size, 256).to(device)

    # model training
    print("Training...")
    train(args, device, g, reverse_eids, seed_edges, model)

    # validate/test the model
    print("Validation/Testing...")
    valid_mrr, test_mrr = evaluate(
        device, g, edge_split, model, batch_size=1000
    )
    print(
        "Validation MRR {:.4f}, Test MRR {:.4f}".format(
            valid_mrr.item(), test_mrr.item()
        )
    )
