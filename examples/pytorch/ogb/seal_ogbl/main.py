import argparse
import math
import os
import random
import sys
import time

import dgl

import numpy as np
import torch
import torch.nn.functional as F
from dgl.dataloading import DataLoader, Sampler
from dgl.nn import GraphConv, SortPooling
from dgl.sampling import global_uniform_negative_sampling
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from scipy.sparse.csgraph import shortest_path
from torch.nn import (
    BCEWithLogitsLoss,
    Conv1d,
    Embedding,
    Linear,
    MaxPool1d,
    ModuleList,
)
from tqdm import tqdm


class Logger(object):
    def __init__(self, runs, info=None):
        self.info = info
        self.results = [[] for _ in range(runs)]

    def add_result(self, run, result):
        # result is in the format of (val_score, test_score)
        assert len(result) == 2
        assert run >= 0 and run < len(self.results)
        self.results[run].append(result)

    def print_statistics(self, run=None, f=sys.stdout):
        if run is not None:
            result = 100 * torch.tensor(self.results[run])
            argmax = result[:, 0].argmax().item()
            print(f"Run {run + 1:02d}:", file=f)
            print(f"Highest Valid: {result[:, 0].max():.2f}", file=f)
            print(f"Highest Eval Point: {argmax + 1}", file=f)
            print(f"   Final Test: {result[argmax, 1]:.2f}", file=f)
        else:
            result = 100 * torch.tensor(self.results)

            best_results = []
            for r in result:
                valid = r[:, 0].max().item()
                test = r[r[:, 0].argmax(), 1].item()
                best_results.append((valid, test))

            best_result = torch.tensor(best_results)

            print(f"All runs:", file=f)
            r = best_result[:, 0]
            print(f"Highest Valid: {r.mean():.2f} ± {r.std():.2f}", file=f)
            r = best_result[:, 1]
            print(f"   Final Test: {r.mean():.2f} ± {r.std():.2f}", file=f)


class SealSampler(Sampler):
    def __init__(
        self,
        g,
        num_hops=1,
        sample_ratio=1.0,
        directed=False,
        prefetch_node_feats=None,
        prefetch_edge_feats=None,
    ):
        super().__init__()
        self.g = g
        self.num_hops = num_hops
        self.sample_ratio = sample_ratio
        self.directed = directed
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats

    def _double_radius_node_labeling(self, adj):
        N = adj.shape[0]
        adj_wo_src = adj[range(1, N), :][:, range(1, N)]
        idx = list(range(1)) + list(range(2, N))
        adj_wo_dst = adj[idx, :][:, idx]

        dist2src = shortest_path(
            adj_wo_dst, directed=False, unweighted=True, indices=0
        )
        dist2src = np.insert(dist2src, 1, 0, axis=0)
        dist2src = torch.from_numpy(dist2src)

        dist2dst = shortest_path(
            adj_wo_src, directed=False, unweighted=True, indices=0
        )
        dist2dst = np.insert(dist2dst, 0, 0, axis=0)
        dist2dst = torch.from_numpy(dist2dst)

        dist = dist2src + dist2dst
        dist_over_2, dist_mod_2 = (
            torch.div(dist, 2, rounding_mode="floor"),
            dist % 2,
        )

        z = 1 + torch.min(dist2src, dist2dst)
        z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
        z[0:2] = 1.0
        # shortest path may include inf values
        z[torch.isnan(z)] = 0.0

        return z.to(torch.long)

    def sample(self, aug_g, seed_edges):
        g = self.g
        subgraphs = []
        # construct k-hop enclosing graph for each link
        for eid in seed_edges:
            src, dst = map(int, aug_g.find_edges(eid))
            # construct the enclosing graph
            visited, nodes, fringe = [np.unique([src, dst]) for _ in range(3)]
            for _ in range(self.num_hops):
                if not self.directed:
                    _, fringe = g.out_edges(fringe)
                else:
                    _, out_neighbors = g.out_edges(fringe)
                    in_neighbors, _ = g.in_edges(fringe)
                    fringe = np.union1d(in_neighbors, out_neighbors)
                fringe = np.setdiff1d(fringe, visited)
                visited = np.union1d(visited, fringe)
                if self.sample_ratio < 1.0:
                    fringe = np.random.choice(
                        fringe,
                        int(self.sample_ratio * len(fringe)),
                        replace=False,
                    )
                if len(fringe) == 0:
                    break
                nodes = np.union1d(nodes, fringe)
            subg = g.subgraph(nodes, store_ids=True)

            # remove edges to predict
            edges_to_remove = [
                subg.edge_ids(s, t)
                for s, t in [(0, 1), (1, 0)]
                if subg.has_edges_between(s, t)
            ]
            subg.remove_edges(edges_to_remove)
            # add double radius node labeling
            subg.ndata["z"] = self._double_radius_node_labeling(
                subg.adj_external(scipy_fmt="csr")
            )
            subg_aug = subg.add_self_loop()
            if "weight" in subg.edata:
                subg_aug.edata["weight"][subg.num_edges() :] = torch.ones(
                    subg_aug.num_edges() - subg.num_edges()
                )
            subgraphs.append(subg_aug)

        subgraphs = dgl.batch(subgraphs)
        dgl.set_src_lazy_features(subg_aug, self.prefetch_node_feats)
        dgl.set_edge_lazy_features(subg_aug, self.prefetch_edge_feats)

        return subgraphs, aug_g.edata["y"][seed_edges]


# An end-to-end deep learning architecture for graph classification, AAAI-18.
class DGCNN(torch.nn.Module):
    def __init__(
        self, hidden_channels, num_layers, k, GNN=GraphConv, feature_dim=0
    ):
        super(DGCNN, self).__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.sort_pool = SortPooling(k=k)

        self.max_z = 1000
        self.z_embedding = Embedding(self.max_z, hidden_channels)

        self.convs = ModuleList()
        initial_channels = hidden_channels + self.feature_dim

        self.convs.append(GNN(initial_channels, hidden_channels))
        for _ in range(0, num_layers - 1):
            self.convs.append(GNN(hidden_channels, hidden_channels))
        self.convs.append(GNN(hidden_channels, 1))

        conv1d_channels = [16, 32]
        total_latent_dim = hidden_channels * num_layers + 1
        conv1d_kws = [total_latent_dim, 5]
        self.conv1 = Conv1d(1, conv1d_channels[0], conv1d_kws[0], conv1d_kws[0])
        self.maxpool1d = MaxPool1d(2, 2)
        self.conv2 = Conv1d(
            conv1d_channels[0], conv1d_channels[1], conv1d_kws[1], 1
        )
        dense_dim = int((self.k - 2) / 2 + 1)
        dense_dim = (dense_dim - conv1d_kws[1] + 1) * conv1d_channels[1]
        self.lin1 = Linear(dense_dim, 128)
        self.lin2 = Linear(128, 1)

    def forward(self, g, z, x=None, edge_weight=None):
        z_emb = self.z_embedding(z)
        if z_emb.ndim == 3:  # in case z has multiple integer labels
            z_emb = z_emb.sum(dim=1)
        if x is not None:
            x = torch.cat([z_emb, x.to(torch.float)], 1)
        else:
            x = z_emb
        xs = [x]

        for conv in self.convs:
            xs += [torch.tanh(conv(g, xs[-1], edge_weight=edge_weight))]
        x = torch.cat(xs[1:], dim=-1)

        # global pooling
        x = self.sort_pool(g, x)
        x = x.unsqueeze(1)  # [num_graphs, 1, k * hidden]
        x = F.relu(self.conv1(x))
        x = self.maxpool1d(x)
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # [num_graphs, dense_dim]

        # MLP.
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


def get_pos_neg_edges(split, split_edge, g, percent=100):
    pos_edge = split_edge[split]["edge"]
    if split == "train":
        neg_edge = torch.stack(
            global_uniform_negative_sampling(
                g, num_samples=pos_edge.size(0), exclude_self_loops=True
            ),
            dim=1,
        )
    else:
        neg_edge = split_edge[split]["edge_neg"]

    # sampling according to the percent param
    np.random.seed(123)
    # pos sampling
    num_pos = pos_edge.size(0)
    perm = np.random.permutation(num_pos)
    perm = perm[: int(percent / 100 * num_pos)]
    pos_edge = pos_edge[perm]
    # neg sampling
    if neg_edge.dim() > 2:  # [Np, Nn, 2]
        neg_edge = neg_edge[perm].view(-1, 2)
    else:
        np.random.seed(123)
        num_neg = neg_edge.size(0)
        perm = np.random.permutation(num_neg)
        perm = perm[: int(percent / 100 * num_neg)]
        neg_edge = neg_edge[perm]

    return pos_edge, neg_edge  # ([2, Np], [2, Nn]) -> ([Np, 2], [Nn, 2])


def train():
    model.train()
    loss_fnt = BCEWithLogitsLoss()
    total_loss = 0
    total = 0
    pbar = tqdm(train_loader, ncols=70)
    for gs, y in pbar:
        optimizer.zero_grad()
        logits = model(
            gs,
            gs.ndata["z"],
            gs.ndata.get("feat", None),
            edge_weight=gs.edata.get("weight", None),
        )
        loss = loss_fnt(logits.view(-1), y.to(torch.float))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * gs.batch_size
        total += gs.batch_size

    return total_loss / total


@torch.no_grad()
def test():
    model.eval()

    y_pred, y_true = [], []
    for gs, y in tqdm(val_loader, ncols=70):
        logits = model(
            gs,
            gs.ndata["z"],
            gs.ndata.get("feat", None),
            edge_weight=gs.edata.get("weight", None),
        )
        y_pred.append(logits.view(-1).cpu())
        y_true.append(y.view(-1).cpu().to(torch.float))
    val_pred, val_true = torch.cat(y_pred), torch.cat(y_true)
    pos_val_pred = val_pred[val_true == 1]
    neg_val_pred = val_pred[val_true == 0]

    y_pred, y_true = [], []
    for gs, y in tqdm(test_loader, ncols=70):
        logits = model(
            gs,
            gs.ndata["z"],
            gs.ndata.get("feat", None),
            edge_weight=gs.edata.get("weight", None),
        )
        y_pred.append(logits.view(-1).cpu())
        y_true.append(y.view(-1).cpu().to(torch.float))
    test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
    pos_test_pred = test_pred[test_true == 1]
    neg_test_pred = test_pred[test_true == 0]

    if args.eval_metric == "hits":
        results = evaluate_hits(
            pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred
        )
    elif args.eval_metric == "mrr":
        results = evaluate_mrr(
            pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred
        )

    return results


def evaluate_hits(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    results = {}
    for K in [20, 50, 100]:
        evaluator.K = K
        valid_hits = evaluator.eval(
            {
                "y_pred_pos": pos_val_pred,
                "y_pred_neg": neg_val_pred,
            }
        )[f"hits@{K}"]
        test_hits = evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )[f"hits@{K}"]

        results[f"Hits@{K}"] = (valid_hits, test_hits)

    return results


def evaluate_mrr(pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    print(
        pos_val_pred.size(),
        neg_val_pred.size(),
        pos_test_pred.size(),
        neg_test_pred.size(),
    )
    neg_val_pred = neg_val_pred.view(pos_val_pred.shape[0], -1)
    neg_test_pred = neg_test_pred.view(pos_test_pred.shape[0], -1)
    results = {}
    valid_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_val_pred,
                "y_pred_neg": neg_val_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    test_mrr = (
        evaluator.eval(
            {
                "y_pred_pos": pos_test_pred,
                "y_pred_neg": neg_test_pred,
            }
        )["mrr_list"]
        .mean()
        .item()
    )

    results["MRR"] = (valid_mrr, test_mrr)

    return results


if __name__ == "__main__":
    # Data settings
    parser = argparse.ArgumentParser(description="OGBL (SEAL)")
    parser.add_argument("--dataset", type=str, default="ogbl-collab")
    # GNN settings
    parser.add_argument("--sortpool_k", type=float, default=0.6)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--hidden_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    # Subgraph extraction settings
    parser.add_argument("--ratio_per_hop", type=float, default=1.0)
    parser.add_argument(
        "--use_feature",
        action="store_true",
        help="whether to use raw node features as GNN input",
    )
    parser.add_argument(
        "--use_edge_weight",
        action="store_true",
        help="whether to consider edge weight in GNN",
    )
    # Training settings
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--train_percent", type=float, default=100)
    parser.add_argument("--val_percent", type=float, default=100)
    parser.add_argument("--test_percent", type=float, default=100)
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="number of workers for dynamic dataloaders",
    )
    # Testing settings
    parser.add_argument("--use_valedges_as_input", action="store_true")
    parser.add_argument("--eval_steps", type=int, default=1)
    args = parser.parse_args()

    data_appendix = "_rph{}".format("".join(str(args.ratio_per_hop).split(".")))
    if args.use_valedges_as_input:
        data_appendix += "_uvai"

    args.res_dir = os.path.join(
        "results/{}_{}".format(args.dataset, time.strftime("%Y%m%d%H%M%S"))
    )
    print("Results will be saved in " + args.res_dir)
    if not os.path.exists(args.res_dir):
        os.makedirs(args.res_dir)
    log_file = os.path.join(args.res_dir, "log.txt")
    # Save command line input.
    cmd_input = "python " + " ".join(sys.argv) + "\n"
    with open(os.path.join(args.res_dir, "cmd_input.txt"), "a") as f:
        f.write(cmd_input)
    print("Command line input: " + cmd_input + " is saved.")
    with open(log_file, "a") as f:
        f.write("\n" + cmd_input)

    dataset = DglLinkPropPredDataset(name=args.dataset)
    split_edge = dataset.get_edge_split()
    graph = dataset[0]

    # re-format the data of citation2
    if args.dataset == "ogbl-citation2":
        for k in ["train", "valid", "test"]:
            src = split_edge[k]["source_node"]
            tgt = split_edge[k]["target_node"]
            split_edge[k]["edge"] = torch.stack([src, tgt], dim=1)
            if k != "train":
                tgt_neg = split_edge[k]["target_node_neg"]
                split_edge[k]["edge_neg"] = torch.stack(
                    [src[:, None].repeat(1, tgt_neg.size(1)), tgt_neg], dim=-1
                )  # [Ns, Nt, 2]

    # reconstruct the graph for ogbl-collab data for validation edge augmentation and coalesce
    if args.dataset == "ogbl-collab":
        graph.edata.pop("year")
        # float edata for to_simple transform
        graph.edata["weight"] = graph.edata["weight"].to(torch.float)
        if args.use_valedges_as_input:
            val_edges = split_edge["valid"]["edge"]
            row, col = val_edges.t()
            val_weights = torch.ones(size=(val_edges.size(0), 1))
            graph.add_edges(
                torch.cat([row, col]),
                torch.cat([col, row]),
                {"weight": val_weights},
            )
        graph = graph.to_simple(copy_edata=True, aggregator="sum")

    if not args.use_edge_weight and "weight" in graph.edata:
        graph.edata.pop("weight")
    if not args.use_feature and "feat" in graph.ndata:
        graph.ndata.pop("feat")

    if args.dataset.startswith("ogbl-citation"):
        args.eval_metric = "mrr"
        directed = True
    else:
        args.eval_metric = "hits"
        directed = False

    evaluator = Evaluator(name=args.dataset)
    if args.eval_metric == "hits":
        loggers = {
            "Hits@20": Logger(args.runs, args),
            "Hits@50": Logger(args.runs, args),
            "Hits@100": Logger(args.runs, args),
        }
    elif args.eval_metric == "mrr":
        loggers = {
            "MRR": Logger(args.runs, args),
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = dataset.root + "_seal{}".format(data_appendix)

    loaders = []
    prefetch_node_feats = ["feat"] if "feat" in graph.ndata else None
    prefetch_edge_feats = ["weight"] if "weight" in graph.edata else None

    train_edge, train_edge_neg = get_pos_neg_edges(
        "train", split_edge, graph, args.train_percent
    )
    val_edge, val_edge_neg = get_pos_neg_edges(
        "valid", split_edge, graph, args.val_percent
    )
    test_edge, test_edge_neg = get_pos_neg_edges(
        "test", split_edge, graph, args.test_percent
    )
    # create an augmented graph for sampling
    aug_g = dgl.graph(graph.edges())
    aug_g.edata["y"] = torch.ones(aug_g.num_edges())
    aug_edges = torch.cat(
        [val_edge, test_edge, train_edge_neg, val_edge_neg, test_edge_neg]
    )
    aug_labels = torch.cat(
        [
            torch.ones(len(val_edge) + len(test_edge)),
            torch.zeros(
                len(train_edge_neg) + len(val_edge_neg) + len(test_edge_neg)
            ),
        ]
    )
    aug_g.add_edges(aug_edges[:, 0], aug_edges[:, 1], {"y": aug_labels})
    # eids for sampling
    split_len = [graph.num_edges()] + list(
        map(
            len,
            [val_edge, test_edge, train_edge_neg, val_edge_neg, test_edge_neg],
        )
    )
    train_eids = torch.cat(
        [
            graph.edge_ids(train_edge[:, 0], train_edge[:, 1]),
            torch.arange(sum(split_len[:3]), sum(split_len[:4])),
        ]
    )
    val_eids = torch.cat(
        [
            torch.arange(sum(split_len[:1]), sum(split_len[:2])),
            torch.arange(sum(split_len[:4]), sum(split_len[:5])),
        ]
    )
    test_eids = torch.cat(
        [
            torch.arange(sum(split_len[:2]), sum(split_len[:3])),
            torch.arange(sum(split_len[:5]), sum(split_len[:6])),
        ]
    )
    sampler = SealSampler(
        graph,
        1,
        args.ratio_per_hop,
        directed,
        prefetch_node_feats,
        prefetch_edge_feats,
    )
    # force to be dynamic for consistent dataloading
    for split, shuffle, eids in zip(
        ["train", "valid", "test"],
        [True, False, False],
        [train_eids, val_eids, test_eids],
    ):
        data_loader = DataLoader(
            aug_g,
            eids,
            sampler,
            shuffle=shuffle,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        loaders.append(data_loader)
    train_loader, val_loader, test_loader = loaders

    # convert sortpool_k from percentile to number.
    num_nodes = []
    for subgs, _ in train_loader:
        subgs = dgl.unbatch(subgs)
        if len(num_nodes) > 1000:
            break
        for subg in subgs:
            num_nodes.append(subg.num_nodes())
    num_nodes = sorted(num_nodes)
    k = num_nodes[int(math.ceil(args.sortpool_k * len(num_nodes))) - 1]
    k = max(k, 10)

    for run in range(args.runs):
        model = DGCNN(
            args.hidden_channels,
            args.num_layers,
            k,
            feature_dim=graph.ndata["feat"].size(1) if args.use_feature else 0,
        ).to(device)
        parameters = list(model.parameters())
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr)
        total_params = sum(p.numel() for param in parameters for p in param)
        print(f"Total number of parameters is {total_params}")
        print(f"SortPooling k is set to {k}")
        with open(log_file, "a") as f:
            print(f"Total number of parameters is {total_params}", file=f)
            print(f"SortPooling k is set to {k}", file=f)

        start_epoch = 1
        # Training starts
        for epoch in range(start_epoch, start_epoch + args.epochs):
            loss = train()

            if epoch % args.eval_steps == 0:
                results = test()
                for key, result in results.items():
                    loggers[key].add_result(run, result)

                model_name = os.path.join(
                    args.res_dir,
                    "run{}_model_checkpoint{}.pth".format(run + 1, epoch),
                )
                optimizer_name = os.path.join(
                    args.res_dir,
                    "run{}_optimizer_checkpoint{}.pth".format(run + 1, epoch),
                )
                torch.save(model.state_dict(), model_name)
                torch.save(optimizer.state_dict(), optimizer_name)

                for key, result in results.items():
                    valid_res, test_res = result
                    to_print = (
                        f"Run: {run + 1:02d}, Epoch: {epoch:02d}, "
                        + f"Loss: {loss:.4f}, Valid: {100 * valid_res:.2f}%, "
                        + f"Test: {100 * test_res:.2f}%"
                    )
                    print(key)
                    print(to_print)
                    with open(log_file, "a") as f:
                        print(key, file=f)
                        print(to_print, file=f)

        for key in loggers.keys():
            print(key)
            loggers[key].print_statistics(run)
            with open(log_file, "a") as f:
                print(key, file=f)
                loggers[key].print_statistics(run, f=f)

    for key in loggers.keys():
        print(key)
        loggers[key].print_statistics()
        with open(log_file, "a") as f:
            print(key, file=f)
            loggers[key].print_statistics(f=f)
    print(f"Total number of parameters is {total_params}")
    print(f"Results are saved in {args.res_dir}")
