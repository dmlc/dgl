import argparse
import os

# For Debug.
import pdb
import time

import dgl
import dgl.graphbolt as gb
import dgl.nn.pytorch as dglnn

import numpy as np

# For memory profiling.
import psutil
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm

# For time profiling.
# from line_profiler import LineProfiler


class SAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            # We need to first copy the representation of nodes on the RHS from
            # the appropriate nodes on the LHS.
            # Note that the shape of h is (num_nodes_LHS, D) and the shape of
            # h_dst would be (num_nodes_RHS, D)
            h_dst = h[: block.num_dst_nodes()]
            # Then we compute the updated representation on the RHS.
            # The shape of h now becomes (num_nodes_RHS, D)
            h = layer(block, (h, h_dst))
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


# @profile
def run(args, device, data):
    # Unpack data.
    train_set, valid_set, test_set, features, num_classes, graph = data

    # Define sampler_func and fetch_func.
    # @profile
    def sampler_func(data):
        adjs = []
        seeds, labels = data[0], data[1]
        fanouts = list(map(int, args.fanouts.split(",")))
        for layer in range(args.num_layers):
            subgraph = graph.sample_neighbors(
                seeds, th.LongTensor([fanouts[layer]])
            )
            subgraph = dgl.graph(subgraph.node_pairs[("_N", "_E", "_N")])
            block = dgl.to_block(subgraph, seeds)
            seeds = block.srcdata[dgl.NID]
            adjs.insert(0, block)

        input_nodes = seeds
        return input_nodes, labels, adjs

    # @profile
    def fetch_func(data):
        input_nodes, labels, adjs = data
        # Get the features of the input nodes.
        input_features = features.read("node", None, "feat", input_nodes)
        # Return the features, labels, and blocks.
        return input_features, labels, adjs

    # Create dgl.graphbolt train dataloader.
    minibatch_sampler = gb.MinibatchSampler(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("finish minibatch_sampler")
    subgraph_sampler = gb.SubgraphSampler(
        minibatch_sampler,
        sampler_func,
    )
    feature_fetcher = gb.FeatureFetcher(subgraph_sampler, fetch_func)
    device_transfer = gb.CopyTo(feature_fetcher, th.device("cpu"))
    dataloader = gb.MultiProcessDataLoader(
        device_transfer, num_workers=args.num_workers
    )
    # dataloader = gb.SingleProcessDataLoader(device_transfer)

    # Create dgl.graphbolt validation/test dataloader.
    validation_minibatch_sampler = gb.MinibatchSampler(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    validation_subgraph_sampler = gb.SubgraphSampler(
        validation_minibatch_sampler,
        sampler_func,
    )
    validation_feature_fetcher = gb.FeatureFetcher(
        validation_subgraph_sampler, fetch_func
    )
    validation_device_transfer = gb.CopyTo(
        validation_feature_fetcher, th.device("cpu")
    )
    validation_dataloader = gb.MultiProcessDataLoader(
        validation_device_transfer, num_workers=args.num_workers
    )

    # Define model and optimizer
    model = SAGE(
        100,
        args.num_hidden,
        num_classes,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device).double()
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # Training loop
    avg = 0
    iter_tput = []
    best_eval_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        model.train()
        # Loop over the dataloader to sample the computation dependency graph as
        # a list of blocks.
        for step, (input_features, batch_labels, blocks) in enumerate(
            dataloader
        ):
            tic_step = time.time()
            # Compute loss and prediction
            batch_pred = model(blocks, input_features)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_tput.append(batch_labels.shape[0] / (time.time() - tic_step))
            if step % args.log_every == 0:
                acc = compute_acc(batch_pred, batch_labels)
                mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else psutil.virtual_memory().used / 1000000
                )
                print(
                    f"Epoch {epoch:05d} | "
                    f"Step {step:05d} | "
                    f"Loss {loss.item():.4f} | "
                    f"Train Acc {acc.item():.4f} | "
                    f"Speed (samples/sec) {np.mean(iter_tput):.4f} | "
                    f"Memo {mem_alloc:.1f} MB",
                    end="\n",
                )

        toc = time.time()
        print("\nEpoch Time(s): {:.4f}".format(toc - tic))
        if epoch >= 5:
            avg += toc - tic
        if epoch % args.eval_every == 0 and epoch != 0:
            ############# Validation ################
            valid_pred = []
            valid_labels = []
            with th.no_grad():
                model.eval()
                for step, (input_features, batch_labels, blocks) in tqdm.tqdm(
                    enumerate(validation_dataloader), desc="Valid"
                ):
                    batch_pred = model(blocks, input_features)
                    valid_pred.append(batch_pred)
                    valid_labels.append(batch_labels)
            valid_pred = th.cat(valid_pred, 0)
            valid_labels = th.cat(valid_labels, 0)
            eval_acc = compute_acc(valid_pred, valid_labels)
            if eval_acc > best_eval_acc:
                best_eval_acc = eval_acc
            print(
                "Eval Acc {:.4f} Best Eval Acc {:.4f} ".format(
                    eval_acc,
                    best_eval_acc,
                )
            )

    ############# Test ################
    # Change the dataloader to the test set.
    test_minibatch_sampler = gb.MinibatchSampler(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_subgraph_sampler = gb.SubgraphSampler(
        test_minibatch_sampler,
        sampler_func,
    )
    test_feature_fetcher = gb.FeatureFetcher(test_subgraph_sampler, fetch_func)
    test_device_transfer = gb.CopyTo(test_feature_fetcher, th.device("cpu"))
    test_dataloader = gb.MultiProcessDataLoader(
        test_device_transfer, num_workers=args.num_workers
    )

    test_pred = []
    test_labels = []
    with th.no_grad():
        model.eval()
        for step, (input_features, batch_labels, blocks) in tqdm.tqdm(
            enumerate(test_dataloader), desc="Test"
        ):
            batch_pred = model(blocks, input_features)
            test_pred.append(batch_pred)
            test_labels.append(batch_labels)
    test_pred = th.cat(test_pred, 0)
    test_labels = th.cat(test_labels, 0)
    test_acc = compute_acc(test_pred, test_labels)

    print("Avg epoch time: {}".format(avg / (epoch - 4)))
    return test_acc


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device ID. Use -1 for CPU training",
    )
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=256)
    argparser.add_argument("--num-layers", type=int, default=3)
    argparser.add_argument("--fanouts", type=str, default="5,10,15")
    argparser.add_argument("--batch-size", type=int, default=1000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=1)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument("--num_runs", type=int, default=1)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument("--wd", type=float, default=0)
    args = argparser.parse_args()

    if args.gpu >= 0:
        device = th.device("cuda:%d" % args.gpu)
    else:
        device = th.device("cpu")

    # load ogbn-products data
    dataset = gb.OnDiskDataset("../example_ogbn_products/")
    raise NotImplementedError(
        "Please use your absolute path to the dataset."
        "And Delete this line if you have done so."
    )

    features = dataset.feature
    labels = dataset.feature.read("node", None, "label")
    graph = dataset.graph
    train_set = dataset.train_set
    test_set = dataset.test_set
    valid_set = dataset.validation_set

    num_classes = len(th.unique(labels))
    # Pack data
    data = (
        train_set,
        valid_set,
        test_set,
        features,
        num_classes,
        graph,
    )

    # Run 1 times
    test_accs = []
    for i in range(args.num_runs):
        test_accs.append(run(args, device, data).cpu().numpy())
        print(
            "Average test accuracy:", np.mean(test_accs), "±", np.std(test_accs)
        )
