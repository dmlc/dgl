import argparse
import math

import dgl

import torch
import torch.nn.functional as F
from dgl.dataloading.negative_sampler import GlobalUniform
from dgl.nn.pytorch import GraphConv, SAGEConv
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from torch.nn import Linear
from torch.utils.data import DataLoader


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        assert len(result) == 3
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 1].argmax().item()
            print(f"Run {run + 1:02d}:")
            print(f"Highest Train: {result[:, 0].max():.2f}")
            print(f"Highest Valid: {result[:, 1].max():.2f}")
            print(f"  Final Train: {result[argmax, 0]:.2f}")
            print(f"   Final Test: {result[argmax, 2]:.2f}")
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                train1 = r[:, 0].max().item()
                valid = r[:, 1].max().item()
                train2 = r[r[:, 1].argmax(), 0].item()
                test = r[r[:, 1].argmax(), 2].item()
                best_results.append((train1, valid, train2, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:")
            r = best_result[:, 0]
            print(f"Highest Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 1]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 2]
            print(f"  Final Train: {r.mean():.2f} ± {r.std():.2f}")
            r = best_result[:, 3]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}")


class NGNN_GCNConv(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_nonl_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = GraphConv(in_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        ngnn_type,
        dataset,
    ):
        super(GCN, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer
        if ngnn_type == "input":
            self.convs.append(
                NGNN_GCNConv(
                    in_channels,
                    hidden_channels,
                    hidden_channels,
                    num_nonl_layers,
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(GraphConv(hidden_channels, hidden_channels))
        elif ngnn_type == "hidden":
            self.convs.append(GraphConv(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(
                    NGNN_GCNConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        num_nonl_layers,
                    )
                )

        self.convs.append(GraphConv(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class NGNN_SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_nonl_layers,
        *,
        reduce,
    ):
        super(NGNN_SAGEConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = SAGEConv(in_channels, hidden_channels, reduce)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x


class SAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        dropout,
        ngnn_type,
        dataset,
        reduce="mean",
    ):
        super(SAGE, self).__init__()

        self.dataset = dataset
        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer
        if ngnn_type == "input":
            self.convs.append(
                NGNN_SAGEConv(
                    in_channels,
                    hidden_channels,
                    hidden_channels,
                    num_nonl_layers,
                    reduce=reduce,
                )
            )
            for _ in range(num_layers - 2):
                self.convs.append(
                    SAGEConv(hidden_channels, hidden_channels, reduce)
                )
        elif ngnn_type == "hidden":
            self.convs.append(SAGEConv(in_channels, hidden_channels, reduce))
            for _ in range(num_layers - 2):
                self.convs.append(
                    NGNN_SAGEConv(
                        hidden_channels,
                        hidden_channels,
                        hidden_channels,
                        num_nonl_layers,
                        reduce=reduce,
                    )
                )

        self.convs.append(SAGEConv(hidden_channels, out_channels, reduce))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, g, x):
        for conv in self.convs[:-1]:
            x = conv(g, x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](g, x)
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_layers, dropout
    ):
        super(LinkPredictor, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.lins.append(Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, x_i, x_j):
        x = x_i * x_j
        for lin in self.lins[:-1]:
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.sigmoid(x)


def train(model, predictor, g, x, split_edge, optimizer, batch_size):
    model.train()
    predictor.train()

    pos_train_edge = split_edge["train"]["edge"].to(x.device)
    neg_sampler = GlobalUniform(1)
    total_loss = total_examples = 0
    for perm in DataLoader(
        range(pos_train_edge.size(0)), batch_size, shuffle=True
    ):
        optimizer.zero_grad()

        h = model(g, x)

        edge = pos_train_edge[perm].t()

        pos_out = predictor(h[edge[0]], h[edge[1]])
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        edge = neg_sampler(g, edge[0])

        neg_out = predictor(h[edge[0]], h[edge[1]])
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        if model.dataset == "ogbl-ddi":
            torch.nn.utils.clip_grad_norm_(x, 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, g, x, split_edge, evaluator, batch_size):
    model.eval()
    predictor.eval()

    h = model(g, x)

    pos_train_edge = split_edge["eval_train"]["edge"].to(h.device)
    pos_valid_edge = split_edge["valid"]["edge"].to(h.device)
    neg_valid_edge = split_edge["valid"]["edge_neg"].to(h.device)
    pos_test_edge = split_edge["test"]["edge"].to(h.device)
    neg_test_edge = split_edge["test"]["edge_neg"].to(h.device)

    def get_pred(test_edges, h):
        preds = []
        for perm in DataLoader(range(test_edges.size(0)), batch_size):
            edge = test_edges[perm].t()
            preds += [predictor(h[edge[0]], h[edge[1]]).squeeze().cpu()]
        pred = torch.cat(preds, dim=0)
        return pred

    pos_train_pred = get_pred(pos_train_edge, h)
    pos_valid_pred = get_pred(pos_valid_edge, h)
    neg_valid_pred = get_pred(neg_valid_edge, h)
    pos_test_pred = get_pred(pos_test_edge, h)
    neg_test_pred = get_pred(neg_test_edge, h)

    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        train_hits = evaluator.eval(
            {
                "y_pred_pos": pos_train_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_valid_pred,
                "y_pred_neg": neg_valid_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (train_hits, valid_hits, test_hits)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="OGBL(Full Batch GCN/GraphSage + NGNN)"
    )

    # dataset setting
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbl-ddi",
        choices=["ogbl-ddi", "ogbl-collab", "ogbl-ppa"],
    )

    # device setting
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training.",
    )

    # model structure settings
    parser.add_argument(
        "--use_sage",
        action="store_true",
        help="If not set, use GCN by default.",
    )
    parser.add_argument(
        "--ngnn_type",
        type=str,
        default="input",
        choices=["input", "hidden"],
        help="You can set this value from 'input' or 'hidden' to apply NGNN to different GNN layers.",
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="number of GNN layers"
    )
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=64 * 1024)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=400)

    # training settings
    parser.add_argument("--eval_steps", type=int, default=1)
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = (
        f"cuda:{args.device}"
        if args.device != -1 and torch.cuda.is_available()
        else "cpu"
    )
    device = torch.device(device)

    dataset = DglLinkPropPredDataset(name=args.dataset)
    g = dataset[0]
    split_edge = dataset.get_edge_split()

    # We randomly pick some training samples that we want to evaluate on:
    idx = torch.randperm(split_edge["train"]["edge"].size(0))
    idx = idx[: split_edge["valid"]["edge"].size(0)]
    split_edge["eval_train"] = {"edge": split_edge["train"]["edge"][idx]}

    if dataset.name == "ogbl-ppa":
        g.ndata["feat"] = g.ndata["feat"].to(torch.float)

    if dataset.name == "ogbl-ddi":
        emb = torch.nn.Embedding(g.num_nodes(), args.hidden_channels).to(device)
        in_channels = args.hidden_channels
    else:  # ogbl-collab, ogbl-ppa
        in_channels = g.ndata["feat"].size(-1)

    # select model
    if args.use_sage:
        model = SAGE(
            in_channels,
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
            args.ngnn_type,
            dataset.name,
        )
    else:  # GCN
        g = dgl.add_self_loop(g)
        model = GCN(
            in_channels,
            args.hidden_channels,
            args.hidden_channels,
            args.num_layers,
            args.dropout,
            args.ngnn_type,
            dataset.name,
        )

    predictor = LinkPredictor(
        args.hidden_channels, args.hidden_channels, 1, 3, args.dropout
    )

    g, model, predictor = map(lambda x: x.to(device), (g, model, predictor))

    evaluator = Evaluator(name=dataset.name)
    loggers = {
        "Hits@20": Logger(args.runs, args),
        "Hits@50": Logger(args.runs, args),
        "Hits@100": Logger(args.runs, args),
    }

    for run in range(args.runs):
        model.reset_parameters()
        predictor.reset_parameters()
        if dataset.name == "ogbl-ddi":
            torch.nn.init.xavier_uniform_(emb.weight)
            g.ndata["feat"] = emb.weight
        optimizer = torch.optim.Adam(
            list(model.parameters())
            + list(predictor.parameters())
            + (list(emb.parameters()) if dataset.name == "ogbl-ddi" else []),
            lr=args.lr,
        )
        for epoch in range(1, 1 + args.epochs):
            loss = train(
                model,
                predictor,
                g,
                g.ndata["feat"],
                split_edge,
                optimizer,
                args.batch_size,
            )

            if epoch % args.eval_steps == 0:
                results = test(
                    model,
                    predictor,
                    g,
                    g.ndata["feat"],
                    split_edge,
                    evaluator,
                    args.batch_size,
                )
                for key, result in results.items():
                    loggers[key].add_result(run, result)
                    train_hits, valid_hits, test_hits = result
                    print(key)
                    print(
                        f"Run: {run + 1:02d}, "
                        f"Epoch: {epoch:02d}, "
                        f"Loss: {loss:.4f}, "
                        f"Train: {100 * train_hits:.2f}%, "
                        f"Valid: {100 * valid_hits:.2f}%, "
                        f"Test: {100 * test_hits:.2f}%"
                    )
                print("---")

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()


if __name__ == "__main__":
    main()
