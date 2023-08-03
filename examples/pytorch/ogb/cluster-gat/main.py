import argparse
import time
from functools import partial

import dgl
import dgl.nn.pytorch as dglnn

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from ogb.nodeproppred import DglNodePropPredDataset
from sampler import ClusterIter, subgraph_collate_fn
from torch.utils.data import DataLoader


class GAT(nn.Module):
    def __init__(
        self,
        in_feats,
        num_heads,
        n_hidden,
        n_classes,
        n_layers,
        activation,
        dropout=0.0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.num_heads = num_heads
        self.layers.append(
            dglnn.GATConv(
                in_feats,
                n_hidden,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=activation,
                negative_slope=0.2,
            )
        )
        for i in range(1, n_layers - 1):
            self.layers.append(
                dglnn.GATConv(
                    n_hidden * num_heads,
                    n_hidden,
                    num_heads=num_heads,
                    feat_drop=dropout,
                    attn_drop=dropout,
                    activation=activation,
                    negative_slope=0.2,
                )
            )
        self.layers.append(
            dglnn.GATConv(
                n_hidden * num_heads,
                n_classes,
                num_heads=num_heads,
                feat_drop=dropout,
                attn_drop=dropout,
                activation=None,
                negative_slope=0.2,
            )
        )

    def forward(self, g, x):
        h = x
        for l, conv in enumerate(self.layers):
            h = conv(g, h)
            if l < len(self.layers) - 1:
                h = h.flatten(1)
        h = h.mean(1)
        return h.log_softmax(dim=-1)

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GAT model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        num_heads = self.num_heads
        for l, layer in enumerate(self.layers):
            if l < self.n_layers - 1:
                y = th.zeros(
                    g.num_nodes(),
                    self.n_hidden * num_heads
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            else:
                y = th.zeros(
                    g.num_nodes(),
                    self.n_hidden
                    if l != len(self.layers) - 1
                    else self.n_classes,
                )
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.DataLoader(
                g,
                th.arange(g.num_nodes()),
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
            )

            with dataloader.enable_cpu_affinity():
                for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                    block = blocks[0].int().to(device)
                    h = x[input_nodes].to(device)
                    if l < self.n_layers - 1:
                        h = layer(block, h).flatten(1)
                    else:
                        h = layer(block, h)
                        h = h.mean(1)
                        h = h.log_softmax(dim=-1)

                    y[output_nodes] = h.cpu()
            x = y
        return y


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, nfeat, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, nfeat, batch_size, device)
    model.train()
    labels_cpu = labels.to(th.device("cpu"))
    return (
        compute_acc(pred[val_nid], labels_cpu[val_nid]),
        compute_acc(pred[test_nid], labels_cpu[test_nid]),
        pred,
    )


def model_param_summary(model):
    """Count the model parameters"""
    cnt = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total Params {}".format(cnt))


#### Entry point
def run(args, device, data, nfeat):
    # Unpack data
    (
        train_nid,
        val_nid,
        test_nid,
        in_feats,
        labels,
        n_classes,
        g,
        cluster_iterator,
    ) = data
    labels = labels.to(device)

    # Define model and optimizer
    model = GAT(
        in_feats,
        args.num_heads,
        args.num_hidden,
        n_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model_param_summary(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        iter_load = 0
        iter_far = 0
        iter_back = 0
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_start = time.time()
        for step, cluster in enumerate(cluster_iterator):
            mask = cluster.ndata.pop("train_mask")
            if mask.sum() == 0:
                continue
            cluster.edata.pop(dgl.EID)
            cluster = cluster.int().to(device)
            input_nodes = cluster.ndata[dgl.NID]
            batch_inputs = nfeat[input_nodes]
            batch_labels = labels[input_nodes]
            tic_step = time.time()

            # Compute loss and prediction
            batch_pred = model(cluster, batch_inputs)
            batch_pred = batch_pred[mask]
            batch_labels = batch_labels[mask]
            loss = nn.functional.nll_loss(batch_pred, batch_labels)
            optimizer.zero_grad()
            tic_far = time.time()
            loss.backward()
            optimizer.step()
            tic_back = time.time()
            iter_load += tic_step - tic_start
            iter_far += tic_far - tic_step
            iter_back += tic_back - tic_far

            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                print(
                    "Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB".format(
                        epoch, step, loss.item(), acc.item(), gpu_mem_alloc
                    )
                )
                tic_start = time.time()

        toc = time.time()
        print(
            "Epoch Time(s): {:.4f} Load {:.4f} Forward {:.4f} Backward {:.4f}".format(
                toc - tic, iter_load, iter_far, iter_back
            )
        )
        if epoch >= 5:
            avg += toc - tic

        if epoch % args.eval_every == 0 and epoch != 0:
            eval_acc, test_acc, pred = evaluate(
                model,
                g,
                nfeat,
                labels,
                val_nid,
                test_nid,
                args.val_batch_size,
                device,
            )
            model = model.to(device)
            if args.save_pred:
                np.savetxt(
                    args.save_pred + "%02d" % epoch,
                    pred.argmax(1).cpu().numpy(),
                    "%d",
                )
            print("Eval Acc {:.4f}".format(eval_acc))
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
                best_test_acc = test_acc
            print(
                "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
                    best_eval_acc, best_test_acc
                )
            )

    if epoch >= 5:
        print("Avg epoch time: {}".format(avg / (epoch - 4)))
    return best_test_acc.to(th.device("cpu"))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num_epochs", type=int, default=20)
    argparser.add_argument("--num_hidden", type=int, default=128)
    argparser.add_argument("--num_layers", type=int, default=3)
    argparser.add_argument("--num_heads", type=int, default=8)
    argparser.add_argument("--batch_size", type=int, default=32)
    argparser.add_argument("--val_batch_size", type=int, default=2000)
    argparser.add_argument("--log_every", type=int, default=20)
    argparser.add_argument("--eval_every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--save_pred", type=str, default="")
    argparser.add_argument("--wd", type=float, default=0)
    argparser.add_argument("--num_partitions", type=int, default=15000)
    argparser.add_argument("--num_workers", type=int, default=4)
    argparser.add_argument(
        "--data_cpu",
        action="store_true",
        help="By default the script puts all node features and labels "
        "on GPU when using it to save time for data copy. This may "
        "be undesired if they cannot fit in GPU memory at once. "
        "This flag disables that.",
    )
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    # load ogbn-products data
    data = DglNodePropPredDataset(name="ogbn-products")
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = (
        splitted_idx["train"],
        splitted_idx["valid"],
        splitted_idx["test"],
    )
    graph, labels = data[0]
    labels = labels[:, 0]
    print("Total edges before adding self-loop {}".format(graph.num_edges()))
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    print("Total edges after adding self-loop {}".format(graph.num_edges()))
    num_nodes = train_idx.shape[0] + val_idx.shape[0] + test_idx.shape[0]
    assert num_nodes == graph.num_nodes()
    mask = th.zeros(num_nodes, dtype=th.bool)
    mask[train_idx] = True
    graph.ndata["train_mask"] = mask

    graph.in_degrees(0)
    graph.out_degrees(0)
    graph.find_edges(0)

    cluster_iter_data = ClusterIter(
        "ogbn-products", graph, args.num_partitions, args.batch_size
    )
    cluster_iterator = DataLoader(
        cluster_iter_data,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers,
        collate_fn=partial(subgraph_collate_fn, graph),
    )

    in_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()
    # Pack data
    data = (
        train_idx,
        val_idx,
        test_idx,
        in_feats,
        labels,
        n_classes,
        graph,
        cluster_iterator,
    )

    # Run 10 times
    test_accs = []
    nfeat = graph.ndata.pop("feat").to(device)
    for i in range(10):
        test_accs.append(run(args, device, data, nfeat))

    print("Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs))
