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

v_t = dgl.__version__


def prepare_data(args, device):
    dataset = DglNodePropPredDataset(name="ogbn-mag")
    split_idx = dataset.get_idx_split()
    # graph: dgl graph object, label: torch tensor of shape (num_nodes, num_tasks)
    g, labels = dataset[0]
    labels = labels["paper"].flatten()

    transform = Compose([ToSimple(), AddReverse()])
    g = transform(g)

    print("Loaded graph: {}".format(g))

    logger = Logger(args.runs)

    # train sampler
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

    return g, labels, dataset.num_classes, split_idx, logger, train_loader


def extract_embed(node_embed, input_nodes):
    emb = node_embed(
        {ntype: input_nodes[ntype] for ntype in input_nodes if ntype != "paper"}
    )
    return emb


def rel_graph_embed(graph, embed_size):
    node_num = {}
    for ntype in graph.ntypes:
        if ntype == "paper":
            continue
        node_num[ntype] = graph.num_nodes(ntype)
    embeds = HeteroEmbedding(node_num, embed_size)
    return embeds


class RelGraphConvLayer(nn.Module):
    def __init__(
        self, in_feat, out_feat, ntypes, rel_names, activation=None, dropout=0.0
    ):
        super(RelGraphConvLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.ntypes = ntypes
        self.rel_names = rel_names
        self.activation = activation

        self.conv = dglnn.HeteroGraphConv(
            {
                rel: dglnn.GraphConv(
                    in_feat, out_feat, norm="right", weight=False, bias=False
                )
                for rel in rel_names
            }
        )

        self.weight = nn.ModuleDict(
            {
                rel_name: nn.Linear(in_feat, out_feat, bias=False)
                for rel_name in self.rel_names
            }
        )

        # weight for self loop
        self.loop_weights = nn.ModuleDict(
            {
                ntype: nn.Linear(in_feat, out_feat, bias=True)
                for ntype in self.ntypes
            }
        )

        self.dropout = nn.Dropout(dropout)
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
        g = g.local_var()
        wdict = {
            rel_name: {"weight": self.weight[rel_name].weight.T}
            for rel_name in self.rel_names
        }

        inputs_dst = {
            k: v[: g.number_of_dst_nodes(k)] for k, v in inputs.items()
        }

        hs = self.conv(g, inputs, mod_kwargs=wdict)

        def _apply(ntype, h):
            h = h + self.loop_weights[ntype](inputs_dst[ntype])
            if self.activation:
                h = self.activation(h)
            return self.dropout(h)

        return {ntype: _apply(ntype, h) for ntype, h in hs.items()}


class EntityClassify(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(EntityClassify, self).__init__()
        self.in_dim = in_dim
        self.h_dim = 64
        self.out_dim = out_dim
        self.rel_names = list(set(g.etypes))
        self.rel_names.sort()
        self.dropout = 0.5

        self.layers = nn.ModuleList()
        # i2h
        self.layers.append(
            RelGraphConvLayer(
                self.in_dim,
                self.h_dim,
                g.ntypes,
                self.rel_names,
                activation=F.relu,
                dropout=self.dropout,
            )
        )

        # h2o
        self.layers.append(
            RelGraphConvLayer(
                self.h_dim,
                self.out_dim,
                g.ntypes,
                self.rel_names,
                activation=None,
            )
        )

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, h, blocks):
        for layer, block in zip(self.layers, blocks):
            h = layer(block, h)
        return h


class Logger(object):
    r"""
    This class was taken directly from the PyG implementation and can be found
    here: https://github.com/snap-stanford/ogb/blob/master/examples/nodeproppred/mag/logger.py

    This was done to ensure that performance was measured in precisely the same way
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

            print(f"All runs:")
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

    for epoch in range(3):
        num_train = split_idx["train"][category].shape[0]
        pbar = tqdm(total=num_train)
        pbar.set_description(f"Epoch {epoch:02d}")
        model.train()

        total_loss = 0

        for input_nodes, seeds, blocks in train_loader:
            blocks = [blk.to(device) for blk in blocks]
            seeds = seeds[
                category
            ]  # we only predict the nodes with type "category"
            batch_size = seeds.shape[0]
            input_nodes_indexes = input_nodes["paper"].to(g.device)
            seeds = seeds.to(labels.device)

            emb = extract_embed(node_embed, input_nodes)
            # Add the batch's raw "paper" features
            emb.update({"paper": g.ndata["feat"]["paper"][input_nodes_indexes]})

            emb = {k: e.to(device) for k, e in emb.items()}
            lbl = labels[seeds].to(device)

            optimizer.zero_grad()
            logits = model(emb, blocks)[category]

            y_hat = logits.log_softmax(dim=-1)
            loss = F.nll_loss(y_hat, lbl)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            pbar.update(batch_size)

        pbar.close()
        loss = total_loss / num_train

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
    model.eval()
    category = "paper"
    evaluator = Evaluator(name="ogbn-mag")

    # 2 GNN layers
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    loader = dgl.dataloading.DataLoader(
        g,
        {"paper": th.arange(g.num_nodes("paper"))},
        sampler,
        batch_size=16384,
        shuffle=False,
        num_workers=0,
        device=device,
    )

    pbar = tqdm(total=y_true.size(0))
    pbar.set_description(f"Inference")

    y_hats = list()

    for input_nodes, seeds, blocks in loader:
        blocks = [blk.to(device) for blk in blocks]
        seeds = seeds[
            category
        ]  # we only predict the nodes with type "category"
        batch_size = seeds.shape[0]
        input_nodes_indexes = input_nodes["paper"].to(g.device)

        emb = extract_embed(node_embed, input_nodes)
        # Get the batch's raw "paper" features
        emb.update({"paper": g.ndata["feat"]["paper"][input_nodes_indexes]})
        emb = {k: e.to(device) for k, e in emb.items()}

        logits = model(emb, blocks)[category]
        y_hat = logits.log_softmax(dim=-1).argmax(dim=1, keepdims=True)
        y_hats.append(y_hat.cpu())

        pbar.update(batch_size)

    pbar.close()

    y_pred = th.cat(y_hats, dim=0)
    y_true = th.unsqueeze(y_true, 1)

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


def is_support_affinity(v_t):
    # dgl supports enable_cpu_affinity since 0.9.1
    return v_t >= "0.9.1"


def main(args):
    device = f"cuda:0" if th.cuda.is_available() else "cpu"

    g, labels, num_classes, split_idx, logger, train_loader = prepare_data(
        args, device
    )

    embed_layer = rel_graph_embed(g, 128).to(device)
    model = EntityClassify(g, 128, num_classes).to(device)

    print(
        f"Number of embedding parameters: {sum(p.numel() for p in embed_layer.parameters())}"
    )
    print(
        f"Number of model parameters: {sum(p.numel() for p in model.parameters())}"
    )

    for run in range(args.runs):
        try:
            embed_layer.reset_parameters()
            model.reset_parameters()
        except:
            # old pytorch version doesn't support reset_parameters() API
            pass

        # optimizer
        all_params = itertools.chain(
            model.parameters(), embed_layer.parameters()
        )
        optimizer = th.optim.Adam(all_params, lr=0.01)

        if (
            args.num_workers != 0
            and device == "cpu"
            and is_support_affinity(v_t)
        ):
            expected_max = int(psutil.cpu_count(logical=False))
            if args.num_workers >= expected_max:
                print(
                    f"[ERROR] You specified num_workers are larger than physical cores, please set any number less than {expected_max}",
                    file=sys.stderr,
                )
            with train_loader.enable_cpu_affinity():
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
        else:
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
