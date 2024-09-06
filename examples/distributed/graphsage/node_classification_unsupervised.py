import argparse
import time
from contextlib import contextmanager

import dgl
import dgl.distributed
import dgl.function as fn
import dgl.nn.pytorch as dglnn

import numpy as np
import sklearn.linear_model as lm
import sklearn.metrics as skm
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm


class DistSAGE(nn.Module):
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
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        The inference code is written in a fashion that it could handle any
        number of nodes and layers.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers are
        # repeated. Therefore, we compute the representation of all nodes layer
        # by layer.  The nodes on each layer are of course splitted in batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            # Create sampler
            sampler = dgl.dataloading.NeighborSampler([-1])
            # Create dataloader
            dataloader = dgl.distributed.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


def load_subtensor(g, input_nodes, device):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = g.ndata["features"][input_nodes].to(device)
    return batch_inputs


class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata["h"] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            pos_score = pos_graph.edata["score"]
        with neg_graph.local_scope():
            neg_graph.ndata["h"] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            neg_score = neg_graph.edata["score"]

        score = th.cat([pos_score, neg_score])
        label = th.cat(
            [th.ones_like(pos_score), th.zeros_like(neg_score)]
        ).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss


def generate_emb(model, g, inputs, batch_size, device):
    """
    Generate embeddings for each node
    g : The entire graph.
    inputs : The features of all the nodes.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)

    return pred


def compute_acc(emb, labels, train_nids, val_nids, test_nids):
    """
    Compute the accuracy of prediction given the labels.

    We will fist train a LogisticRegression model using the trained embeddings,
    the training set, validation set and test set is provided as the arguments.

    The final result is predicted by the lr model.

    emb: The pretrained embeddings
    labels: The ground truth
    train_nids: The training set node ids
    val_nids: The validation set node ids
    test_nids: The test set node ids
    """

    emb = emb[np.arange(labels.shape[0])].cpu().numpy()
    train_nids = train_nids.cpu().numpy()
    val_nids = val_nids.cpu().numpy()
    test_nids = test_nids.cpu().numpy()
    labels = labels.cpu().numpy()

    emb = (emb - emb.mean(0, keepdims=True)) / emb.std(0, keepdims=True)
    lr = lm.LogisticRegression(multi_class="multinomial", max_iter=10000)
    lr.fit(emb[train_nids], labels[train_nids])

    pred = lr.predict(emb)
    eval_acc = skm.accuracy_score(labels[val_nids], pred[val_nids])
    test_acc = skm.accuracy_score(labels[test_nids], pred[test_nids])
    return eval_acc, test_acc


def run(args, device, data):
    # Unpack data
    (
        train_eids,
        train_nids,
        in_feats,
        g,
        global_train_nid,
        global_valid_nid,
        global_test_nid,
        labels,
    ) = data
    # Create sampler
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(args.num_negs)
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")]
    )
    # Create dataloader
    exclude = "reverse_id" if args.remove_edge else None
    reverse_eids = th.arange(g.num_edges()) if args.remove_edge else None
    dataloader = dgl.distributed.DistEdgeDataLoader(
        g,
        train_eids,
        sampler,
        negative_sampler=neg_sampler,
        exclude=exclude,
        reverse_eids=reverse_eids,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    # Define model and optimizer
    model = DistSAGE(
        in_feats,
        args.num_hidden,
        args.num_hidden,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            dev_id = g.rank() % args.num_gpus
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[dev_id], output_device=dev_id
            )
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    epoch = 0
    for epoch in range(args.num_epochs):
        num_seeds = 0
        num_inputs = 0

        step_time = []
        sample_t = []
        feat_copy_t = []
        forward_t = []
        backward_t = []
        update_t = []
        iter_tput = []

        start = time.time()
        with model.join():
            # Loop over the dataloader to sample the computation dependency
            # graph as a list of blocks.
            for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(
                dataloader
            ):
                if args.debug:
                    # Verify exclude_edges functionality.
                    for block in blocks:
                        current_eids = block.edata[dgl.EID]
                        seed_eids = pos_graph.edata[dgl.EID]
                        if exclude is None:
                            assert th.any(th.isin(current_eids, seed_eids))
                        elif exclude == "self":
                            assert not th.any(th.isin(current_eids, seed_eids))
                        elif exclude == "reverse_id":
                            assert not th.any(th.isin(current_eids, seed_eids))
                        else:
                            raise ValueError(
                                f"Unsupported exclude type: {exclude}"
                            )
                tic_step = time.time()
                sample_t.append(tic_step - start)

                copy_t = time.time()
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.to(device) for block in blocks]
                batch_inputs = load_subtensor(g, input_nodes, device)
                copy_time = time.time()
                feat_copy_t.append(copy_time - copy_t)

                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                forward_end = time.time()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_t.append(forward_end - copy_time)
                backward_t.append(compute_end - forward_end)

                # Aggregate gradients in multiple nodes.
                optimizer.step()
                update_t.append(time.time() - compute_end)

                pos_edges = pos_graph.num_edges()

                step_t = time.time() - start
                step_time.append(step_t)
                iter_tput.append(pos_edges / step_t)
                num_seeds += pos_edges
                if step % args.log_every == 0:
                    print(
                        "[{}] Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed "
                        "(samples/sec) {:.4f} | time {:.3f}s | sample {:.3f} | "
                        "copy {:.3f} | forward {:.3f} | backward {:.3f} | "
                        "update {:.3f}".format(
                            g.rank(),
                            epoch,
                            step,
                            loss.item(),
                            np.mean(iter_tput[3:]),
                            np.sum(step_time[-args.log_every :]),
                            np.sum(sample_t[-args.log_every :]),
                            np.sum(feat_copy_t[-args.log_every :]),
                            np.sum(forward_t[-args.log_every :]),
                            np.sum(backward_t[-args.log_every :]),
                            np.sum(update_t[-args.log_every :]),
                        )
                    )
                start = time.time()

        print(
            "[{}]Epoch Time(s): {:.4f}, sample: {:.4f}, data copy: {:.4f}, "
            "forward: {:.4f}, backward: {:.4f}, update: {:.4f}, #seeds: {}, "
            "#inputs: {}".format(
                g.rank(),
                np.sum(step_time),
                np.sum(sample_t),
                np.sum(feat_copy_t),
                np.sum(forward_t),
                np.sum(backward_t),
                np.sum(update_t),
                num_seeds,
                num_inputs,
            )
        )
        epoch += 1

    # evaluate the embedding using LogisticRegression
    pred = generate_emb(
        model if args.standalone else model.module,
        g,
        g.ndata["features"],
        args.batch_size_eval,
        device,
    )
    if g.rank() == 0:
        eval_acc, test_acc = compute_acc(
            pred, labels, global_train_nid, global_valid_nid, global_test_nid
        )
        print("eval acc {:.4f}; test acc {:.4f}".format(eval_acc, test_acc))

    # sync for eval and test
    if not args.standalone:
        th.distributed.barrier()

    if not args.standalone:
        g._client.barrier()

        # save features into file
        if g.rank() == 0:
            th.save(pred, "emb.pt")
    else:
        th.save(pred, "emb.pt")


def main(args):
    print("--- Distributed node classification with GraphSAGE unsuperised ---")
    dgl.distributed.initialize(args.ip_config)
    if not args.standalone:
        th.distributed.init_process_group(backend="gloo")
    g = dgl.distributed.DistGraph(args.graph_name, part_config=args.part_config)
    print("rank:", g.rank())
    print("number of edges", g.num_edges())

    train_eids = dgl.distributed.edge_split(
        th.ones((g.num_edges(),), dtype=th.bool),
        g.get_partition_book(),
        force_even=True,
    )
    train_nids = dgl.distributed.node_split(
        th.ones((g.num_nodes(),), dtype=th.bool), g.get_partition_book()
    )
    global_train_nid = th.LongTensor(
        np.nonzero(g.ndata["train_mask"][np.arange(g.num_nodes())])
    )
    global_valid_nid = th.LongTensor(
        np.nonzero(g.ndata["val_mask"][np.arange(g.num_nodes())])
    )
    global_test_nid = th.LongTensor(
        np.nonzero(g.ndata["test_mask"][np.arange(g.num_nodes())])
    )
    labels = g.ndata["labels"][np.arange(g.num_nodes())]
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))

    # Pack data
    in_feats = g.ndata["features"].shape[1]
    global_train_nid = global_train_nid.squeeze()
    global_valid_nid = global_valid_nid.squeeze()
    global_test_nid = global_test_nid.squeeze()
    print("number of train {}".format(global_train_nid.shape[0]))
    print("number of valid {}".format(global_valid_nid.shape[0]))
    print("number of test {}".format(global_test_nid.shape[0]))
    data = (
        train_eids,
        train_nids,
        in_feats,
        g,
        global_train_nid,
        global_valid_nid,
        global_test_nid,
        labels,
    )
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument("--n_classes", type=int, help="the number of classes")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local_rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument("--num_negs", type=int, default=1)
    parser.add_argument(
        "--remove_edge",
        default=False,
        action="store_true",
        help="whether to remove edges during sampling",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="whether to verify functionality of remove edges",
    )
    args = parser.parse_args()
    print(args)
    main(args)
