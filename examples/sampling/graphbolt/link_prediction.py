import argparse

import dgl
import dgl.graphbolt as gb
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from ogb.linkproppred import Evaluator
from torchdata.datapipes.iter import Mapper


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


def to_link_block_train(data):
    """Convert train ItemSet to a LinkPredictionBlock."""
    block = gb.LinkPredictionBlock(node_pair=data)
    return block


def to_link_block_eval(data):
    """Convert eval ItemSet to a LinkPredictionBlock."""
    block = gb.LinkPredictionBlock(node_pair=data[:2], negative_tail=data[2])
    return block


def get_dataloader(args, graph, features, current_set, is_train=True):
    """Get a graphbolt-version dataloader."""
    minibatch_sampler = gb.MinibatchSampler(
        current_set, batch_size=args.batch_size, shuffle=is_train
    )
    data_block_converter = Mapper(
        minibatch_sampler,
        to_link_block_train if is_train else to_link_block_eval,
    )
    format = (
        gb.LinkPredictionEdgeFormat.INDEPENDENT
        if is_train
        else gb.LinkPredictionEdgeFormat.TAIL_CONDITIONED
    )
    negative_sampler = gb.UniformNegativeSampler(
        data_block_converter,
        args.neg_ratio,
        format,
        graph,
    )
    subgraph_sampler = gb.NeighborSampler(negative_sampler, graph, args.fanout)
    feature_keys = [("node", None, "feat")]
    feature_fetcher = gb.FeatureFetcher(
        subgraph_sampler, features, feature_keys
    )
    dataloader = gb.MultiProcessDataLoader(
        gb.CopyTo(feature_fetcher, torch.device("cpu")),
        num_workers=args.num_workers,
    )
    return dataloader


def to_dgl_blocks(sampled_subgraphs):
    """Convert sampled subgraphs to DGL blocks."""
    blocks = [
        dgl.create_block(
            sampled_subgraph.node_pairs,
            num_src_nodes=sampled_subgraph.reverse_column_node_ids.shape[0],
            num_dst_nodes=sampled_subgraph.reverse_row_node_ids.shape[0],
        )
        for sampled_subgraph in range(sampled_subgraphs)
    ]
    return blocks


@torch.no_grad()
def evaluate(args, graph, features, current_set, model):
    evaluator = Evaluator(name="ogbl-citation2")

    # Since we need to evaluate the model, we need to set the number
    # of layers to 1 and the fanout to -1.
    args.fanout = [torch.LongTensor([-1])]
    dataloader = get_dataloader(
        args, graph, features, current_set, is_train=False
    )
    pos_pred = []
    neg_pred = []

    model.eval()
    for step, data in tqdm.tqdm(enumerate(dataloader)):
        # Unpack LinkPredictionBlock.
        compacted_pairs = data.compacted_node_pair
        node_feature = data.node_feature[(None, "feat")].float()
        blocks = to_dgl_blocks(data.sampled_subgraphs)

        # Get the embeddings of the input nodes.
        y = model(blocks, node_feature)
        pos_score = (
            (model.predictor(y[compacted_pairs[0]] * y[compacted_pairs[1]]))
            .squeeze()
            .detach()
        )
        # We need to repeat the negative source nodes to match the number of
        # negative destination nodes.
        neg_src = compacted_pairs[0].repeat_interleave(args.neg_ratio, dim=0)
        neg_score = (
            model.predictor(y[neg_src] * y[data.compacted_negative_tail])
            .squeeze()
            .detach()
        )
        pos_pred.append(pos_score)
        neg_pred.append(neg_score)
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
        for step, data in enumerate(dataloader):
            # Unpack LinkPredictionBlock.
            labels = data.label
            compacted_pairs = data.compacted_node_pair
            node_feature = data.node_feature[(None, "feat")].float()
            blocks = to_dgl_blocks(data.sampled_subgraphs)

            # Get the embeddings of the input nodes.
            y = model(blocks, node_feature.float())
            logits = model.predictor(
                y[compacted_pairs[0]] * y[compacted_pairs[1]]
            ).squeeze()

            # Compute loss.
            loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(
                f"Epoch {epoch:05d} | "
                f"Step {step:05d} | "
                f"Loss {(total_loss) / (step + 1):.4f}",
                end="\n",
            )

        # Evaluate the model.
        print("Validation")
        _, valid_mrr = evaluate(args, graph, features, valid_set, model)
        print(f"Valid MRR {valid_mrr.item():.4f}")


def parse_args():
    parser = argparse.ArgumentParser(description="OGBL-Citation2 (GraphBolt)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0005)
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
        default="cpu",
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
    args.fanout = [
        torch.LongTensor([int(fanout)]) for fanout in args.fanout.split(",")
    ]

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
