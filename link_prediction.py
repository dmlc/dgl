import argparse

import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import Evaluator


############## Subgraph sampler function ##################
class LinkSampler:
    def __init__(self, graph, num_layers, fanout, neg_ratio=1000):
        assert num_layers == len(fanout)
        self.graph = graph
        self.num_layers = num_layers
        self.fanout = fanout
        self.neg_ratio = neg_ratio

    def __call__(self, data):
        adjs = []
        # Data format is (u, v, neg_v...).
        u = data[0]
        u = torch.cat([u.repeat_interleave(self.neg_ratio), u])
        v = [t.view(1, -1) for t in data[2:]]
        v = torch.cat(v, dim=1).view(-1)
        # Cat pos and neg v.
        v = torch.cat([data[1], v])
        node_pairs = (u, v)

        seeds, compacted_pairs = gb.unique_and_compact_node_pairs(node_pairs)
        for hop in range(self.num_layers):
            sg = self._generate(seeds, torch.LongTensor(self.fanout[hop]))
            sg = dgl.graph(sg.node_pairs[("_N", "_E", "_N")])
            block = dgl.to_block(sg, seeds)
            seeds = block.srcdata[dgl.NID]
            adjs.insert(0, block)

        input_nodes = seeds
        return input_nodes, compacted_pairs, adjs

    def _generate(self, seeds, fanout):
        raise NotImplementedError


class LinkNeighborSampler(LinkSampler):
    def _generate(self, seeds, fanout):
        return self.graph.sample_neighbors(seeds, fanout)


############## Negative sampler function ##################
class NegativeSampler:
    def __init__(self, graph, negative_ratio):
        self.graph = graph
        self.negative_ratio = negative_ratio

    def __call__(self, data):
        node_pairs = data
        neg_pairs = self._generate(node_pairs)
        num_sample = neg_pairs[0].shape[0] // self.negative_ratio
        neg_dst = list(neg_pairs[1].split(num_sample))
        data = list(data) + neg_dst
        return data

    def _generate(self, node_pairs, edge_type=None):
        raise NotImplementedError


class PerSourceUniformSampler(NegativeSampler):
    def _generate(self, node_pairs, edge_type=None):
        return self.graph.sample_negative_edges_uniform(
            edge_type=edge_type,
            node_pairs=node_pairs,
            negative_ratio=self.negative_ratio,
        )


class SAGE(nn.Module):
    def __init__(self, in_size, hidden_size):
        super().__init__()
        self.layers = nn.ModuleList()
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

    def forward(self, blocks, x):
        hidden_x = x
        for layer_idx, (layer, block) in enumerate(zip(self.layers, blocks)):
            hidden_x = layer(block, hidden_x)
            is_last_layer = layer_idx == len(self.layers) - 1
            if not is_last_layer:
                hidden_x = F.relu(hidden_x)
        return hidden_x


def create_fetch_func(features):
    """Create a fetch function."""

    def fetch_func(data):
        """Fetch data from disk."""
        input_nodes, compacted_pairs, adjs = data
        input_features = features.read("node", None, "feat", input_nodes)
        return input_features.float(), compacted_pairs, adjs

    return fetch_func


def get_dataloader(args, graph, features, current_set):
    """Get a graphbolt-version dataloader."""
    #! Why we need so many steps to get a dataloader?
    minibatch_sampler = gb.MinibatchSampler(
        current_set, batch_size=args.batch_size, shuffle=True
    )
    negative_sampler = gb.SubgraphSampler(
        minibatch_sampler,
        PerSourceUniformSampler(graph, args.neg_ratio),
    )
    subgraph_sampler = gb.SubgraphSampler(
        negative_sampler,
        LinkNeighborSampler(graph, args.num_layers, args.fanout),
    )
    fetch_func = create_fetch_func(features)
    feature_fetcher = gb.FeatureFetcher(subgraph_sampler, fetch_func)
    device_transfer = gb.CopyTo(feature_fetcher, torch.device("cpu"))
    dataloader = gb.MultiProcessDataLoader(
        device_transfer, num_workers=args.num_workers
    )
    return dataloader


@torch.no_grad()
def evaluate(args, graph, features, current_set, model):
    evaluator = Evaluator(name="ogbl-citation2")

    # Since we need to evaluate the model, we need to set the number
    # of layers to 1 and the fanout to -1.
    args.num_layers = 1
    args.fanout = [[-1]]
    dataloader = get_dataloader(args, graph, features, current_set)
    pos_pred = []
    neg_pred = []

    model.eval()
    for step, (input_features, compacted_pairs, blocks) in enumerate(
        dataloader
    ):
        y = model(blocks, input_features)
        score = (
            (model.predictor(y[compacted_pairs[0]] * y[compacted_pairs[1]]))
            .squeeze()
            .detach()
        )
        num_batch = compacted_pairs[0].shape[0] // (args.neg_ratio + 1)
        pos_pred.append(score[:num_batch])
        neg_pred.append(score[num_batch:])

    pos_pred = torch.cat(pos_pred, dim=0)
    neg_pred = torch.cat(neg_pred, dim=0).view(-1, args.neg_ratio)

    input_dict = {"y_pred_pos": pos_pred, "y_pred_neg": neg_pred}
    mrr = evaluator.eval(input_dict)["mrr_list"]
    return mrr, mrr.mean()


def train(args, graph, features, train_set, valid_set, model):
    total_loss = 0.0
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    dataloader = get_dataloader(args, graph, features, train_set)

    for epoch in tqdm.trange(args.epochs):
        model.train()
        for step, (input_features, compacted_pairs, blocks) in enumerate(
            dataloader
        ):
            # Get the embeddings of the input nodes.
            y = model(blocks, input_features)
            logits = model.predictor(
                y[compacted_pairs[0]] * y[compacted_pairs[1]]
            ).squeeze()

            # Construct the labels.
            labels = torch.zeros_like(logits)
            num_batch = compacted_pairs[0].shape[0] // (args.neg_ratio + 1)
            labels[:num_batch] = 1.0

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(
                f"Epoch {epoch:05d} | "
                f"Step {step:05d} | "
                f"Loss {loss.item():.4f}",
                end="\r",
            )

        # Evaluate the model.
        print("Validation")
        valid_mrr = evaluate(args, graph, features, valid_set, model)
        print(f"Valid MRR {valid_mrr.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="OGBL-Citation2 (GraphBolt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--neg-ratio", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--fanout",
        type=str,
        default="15,10,5",
        help="Fan-out of neighbor sampling. Default: 15,10,5",
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
    dataset = gb.OnDiskDataset(
        "/home/ubuntu/wklwork/work#4/example_ogbl_citation2"
    ).load()
    graph = dataset.graph
    features = dataset.feature
    train_set = dataset.tasks[0].train_set
    valid_set = dataset.tasks[0].validation_set
    # CSCSamplingGraph can't be used in GPU mode.
    args.fanout = [[int(fanout)] for fanout in args.fanout.split(",")]

    in_size = 128
    hidden_channels = 256
    model = SAGE(in_size, hidden_channels)

    # Model training.
    print("Training...")
    train(args, graph, features, train_set, valid_set, model)

    # Test the model.
    print("Testing...")
    test_set = dataset.tasks[0].test_set
    test_mrr = evaluate(args, graph, features, test_set, model)
    print(f"Test MRR {test_mrr.item():.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
