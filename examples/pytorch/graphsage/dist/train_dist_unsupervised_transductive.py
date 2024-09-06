import argparse
import time

import dgl
import dgl.distributed
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from train_dist_transductive import DistEmb, load_embs
from train_dist_unsupervised import compute_acc, CrossEntropyLoss, DistSAGE


def generate_emb(standalone, model, emb_layer, g, batch_size, device):
    """
    Generate embeddings for each node
    emb_layer : Embedding layer
    g : The entire graph.
    inputs : The features of all the nodes.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    if not standalone:
        model = model.module
    model.eval()
    emb_layer.eval()
    with th.no_grad():
        inputs = load_embs(standalone, emb_layer, g)
        pred = model.inference(g, inputs, batch_size, device)
    g.barrier()
    return pred


def run(args, device, data):
    # Unpack data
    (
        train_eids,
        train_nids,
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
    emb_layer = DistEmb(
        g.num_nodes(),
        args.num_hidden,
        dgl_sparse_emb=args.dgl_sparse,
        dev_id=device,
    )
    model = DistSAGE(
        args.num_hidden,
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
            if not args.dgl_sparse:
                emb_layer = th.nn.parallel.DistributedDataParallel(emb_layer)
    loss_fcn = CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.dgl_sparse:
        emb_optimizer = dgl.distributed.optim.SparseAdam(
            [emb_layer.sparse_emb], lr=args.sparse_lr
        )
        print("optimize DGL sparse embedding:", emb_layer.sparse_emb)
    elif args.standalone:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print("optimize Pytorch sparse embedding:", emb_layer.sparse_emb)
    else:
        emb_optimizer = th.optim.SparseAdam(
            list(emb_layer.module.sparse_emb.parameters()), lr=args.sparse_lr
        )
        print("optimize Pytorch sparse embedding:", emb_layer.module.sparse_emb)

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
                tic_step = time.time()
                sample_t.append(tic_step - start)

                copy_t = time.time()
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.to(device) for block in blocks]
                feat_copy_t.append(copy_t - tic_step)
                copy_time = time.time()

                # Compute loss and prediction
                batch_inputs = emb_layer(input_nodes)
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                forward_end = time.time()
                emb_optimizer.zero_grad()
                optimizer.zero_grad()
                loss.backward()
                compute_end = time.time()
                forward_t.append(forward_end - copy_time)
                backward_t.append(compute_end - forward_end)

                # Aggregate gradients in multiple nodes.
                emb_optimizer.step()
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
        args.standalone, model, emb_layer, g, args.batch_size_eval, device
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
    global_train_nid = global_train_nid.squeeze()
    global_valid_nid = global_valid_nid.squeeze()
    global_test_nid = global_test_nid.squeeze()
    print("number of train {}".format(global_train_nid.shape[0]))
    print("number of valid {}".format(global_valid_nid.shape[0]))
    print("number of test {}".format(global_test_nid.shape[0]))
    data = (
        train_eids,
        train_nids,
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
    parser.add_argument("--num_epochs", type=int, default=5)
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
        "--neg_share",
        default=False,
        action="store_true",
        help="sharing neg nodes for positive nodes",
    )
    parser.add_argument(
        "--remove_edge",
        default=False,
        action="store_true",
        help="whether to remove edges during sampling",
    )
    parser.add_argument(
        "--dgl_sparse",
        action="store_true",
        help="Whether to use DGL sparse embedding",
    )
    parser.add_argument(
        "--sparse_lr", type=float, default=1e-2, help="sparse lr rate"
    )
    args = parser.parse_args()
    print(args)
    main(args)
