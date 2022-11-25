import argparse
import os
import sys
import time

import numpy as np
import torch as th
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import SAGE
from model import compute_acc_unsupervised as compute_acc
from negative_sampler import NegativeSampler
from torch.nn.parallel import DistributedDataParallel

import dgl
import dgl.function as fn

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from load_graph import load_ogb, load_reddit

from dgl.storages import GPUCachedTensorStorage


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


def evaluate(
    model, g, nfeat, labels, train_nids, val_nids, test_nids, device, args
):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            pred = model.inference(
                g, nfeat, device, args.batch_size, args.num_workers
            )
        # multi gpu
        else:
            pred = model.module.inference(
                g, nfeat, device, args.batch_size, args.num_workers
            )
    model.train()
    return compute_acc(pred, labels, train_nids, val_nids, test_nids)


#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = th.device(devices[proc_id])
    if n_gpus > 0:
        th.cuda.set_device(device)
    if n_gpus > 1:
        dist_init_method = "tcp://{master_ip}:{master_port}".format(
            master_ip="127.0.0.1", master_port="12345"
        )
        world_size = n_gpus
        th.distributed.init_process_group(
            backend="nccl",
            init_method=dist_init_method,
            world_size=world_size,
            rank=proc_id,
        )
    train_nid, val_nid, test_nid, n_classes, g, nfeat, labels = data

    in_feats = nfeat.shape[1]

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    if args.graph_device == "gpu":
        train_seeds = train_seeds.to(device)
        g = g.to(device)
        args.num_workers = 0
    elif args.graph_device == "uva":
        train_seeds = train_seeds.to(device)
        g.pin_memory_()
        args.num_workers = 0

    # Create sampler
    sampler = dgl.dataloading.NeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(",")],
        prefetch_node_feats=["features"],
    )
    sampler = dgl.dataloading.as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat(
            [th.arange(n_edges // 2, n_edges), th.arange(0, n_edges // 2)]
        ).to(train_seeds),
        negative_sampler=NegativeSampler(
            g,
            args.num_negs,
            args.neg_share,
            device if args.graph_device == "uva" else None,
        ),
    )
    dataloader = dgl.dataloading.EdgeDataLoader(
        g,
        train_seeds,
        sampler,
        device=device,
        use_ddp=n_gpus > 1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        use_uva=args.graph_device == "uva",
    )

    dataloader.attach_ndata(
        "features", GPUCachedTensorStorage(nfeat, args.cache_size)
    )

    # Define model and optimizer
    model = SAGE(
        in_feats,
        args.num_hidden,
        args.num_hidden,
        args.num_layers,
        F.relu,
        args.dropout,
    )
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device
        )
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    avg = 0
    iter_pos = []
    iter_neg = []
    iter_d = []
    iter_t = []
    best_eval_acc = 0
    best_test_acc = 0
    for epoch in range(args.num_epochs):
        tic = time.time()

        # Loop over the dataloader to sample the computation dependency graph as a list of
        # blocks.
        tic_step = time.time()
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(
            dataloader
        ):
            input_nodes = input_nodes.to(device)
            batch_inputs = blocks[0].srcdata["features"]
            blocks = [block.int() for block in blocks]
            d_step = time.time()

            # Compute loss and prediction
            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, pos_graph, neg_graph)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t = time.time()
            pos_edges = pos_graph.num_edges()
            neg_edges = neg_graph.num_edges()
            iter_pos.append(pos_edges / (t - tic_step))
            iter_neg.append(neg_edges / (t - tic_step))
            iter_d.append(d_step - tic_step)
            iter_t.append(t - d_step)
            if step % args.log_every == 0 and proc_id == 0:
                gpu_mem_alloc = (
                    th.cuda.max_memory_allocated() / 1000000
                    if th.cuda.is_available()
                    else 0
                )
                print(
                    "[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f} | GPU {:.1f} MB".format(
                        proc_id,
                        epoch,
                        step,
                        loss.item(),
                        np.mean(iter_pos[3:]),
                        np.mean(iter_neg[3:]),
                        np.mean(iter_d[3:]),
                        np.mean(iter_t[3:]),
                        gpu_mem_alloc,
                    )
                )
            tic_step = time.time()

        toc = time.time()
        if proc_id == 0:
            print("Epoch Time(s): {:.4f}".format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if (epoch + 1) % args.eval_every == 0:
                eval_acc, test_acc = evaluate(
                    model,
                    g,
                    nfeat,
                    labels,
                    train_nid,
                    val_nid,
                    test_nid,
                    device,
                    args,
                )
                print(
                    "Eval Acc {:.4f} Test Acc {:.4f}".format(eval_acc, test_acc)
                )
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    best_test_acc = test_acc
                print(
                    "Best Eval Acc {:.4f} Test Acc {:.4f}".format(
                        best_eval_acc, best_test_acc
                    )
                )

        if n_gpus > 1:
            th.distributed.barrier()

    if proc_id == 0:
        print("Avg epoch time: {}".format(avg / (epoch - 4)))


def main(args):
    devices = list(map(int, args.gpu.split(",")))
    n_gpus = len(devices)

    # load dataset
    if args.dataset == "reddit":
        g, n_classes = load_reddit(self_loop=False)
    elif args.dataset == "ogbn-products":
        g, n_classes = load_ogb("ogbn-products")
    else:
        raise Exception("unknown dataset")

    train_nid = g.ndata.pop("train_mask").nonzero().squeeze()
    val_nid = g.ndata.pop("val_mask").nonzero().squeeze()
    test_nid = g.ndata.pop("test_mask").nonzero().squeeze()

    nfeat = g.ndata.pop("features")
    labels = g.ndata.pop("labels")

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    g.create_formats_()

    # this to avoid competition overhead on machines with many cores.
    # Change it to a proper number on your machine, especially for multi-GPU training.
    os.environ["OMP_NUM_THREADS"] = str(mp.cpu_count() // 2 // n_gpus)

    # Pack data
    data = train_nid, val_nid, test_nid, n_classes, g, nfeat, labels

    if devices[0] == -1:
        assert (
            args.graph_device == "cpu"
        ), f"Must have GPUs to enable {args.graph_device} sampling."
        assert (
            args.data_device == "cpu"
        ), f"Must have GPUs to enable {args.data_device} feature storage."
        run(0, 0, args, ["cpu"], data)
    elif n_gpus == 1:
        run(0, n_gpus, args, devices, data)
    else:
        mp.spawn(run, args=(n_gpus, args, devices, data), nprocs=n_gpus)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument(
        "--gpu",
        type=str,
        default="0",
        help="GPU, can be a list of gpus for multi-gpu training,"
        " e.g., 0,1,2,3; -1 for CPU",
    )
    argparser.add_argument(
        "--dataset",
        type=str,
        default="reddit",
        choices=("reddit", "ogbn-products"),
    )
    argparser.add_argument("--num-epochs", type=int, default=20)
    argparser.add_argument("--num-hidden", type=int, default=16)
    argparser.add_argument("--num-layers", type=int, default=2)
    argparser.add_argument("--num-negs", type=int, default=1)
    argparser.add_argument(
        "--neg-share",
        default=False,
        action="store_true",
        help="sharing neg nodes for positive nodes",
    )
    argparser.add_argument("--fan-out", type=str, default="10,25")
    argparser.add_argument("--batch-size", type=int, default=10000)
    argparser.add_argument("--log-every", type=int, default=20)
    argparser.add_argument("--eval-every", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.003)
    argparser.add_argument("--dropout", type=float, default=0.5)
    argparser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of sampling processes. Use 0 for no extra process.",
    )
    argparser.add_argument(
        "--graph-device",
        choices=("cpu", "gpu", "uva"),
        default="cpu",
        help="Device to perform the sampling. "
        "Must have 0 workers for 'gpu' and 'uva'",
    )
    argparser.add_argument(
        "--data-device",
        choices=("cpu", "gpu", "uva"),
        default="gpu",
        help="By default the script puts all node features and labels "
        "on GPU when using it to save time for data copy. This may "
        "be undesired if they cannot fit in GPU memory at once. "
        "Use 'cpu' to keep the features on host memory and "
        "'uva' to enable UnifiedTensor (GPU zero-copy access on "
        "pinned host memory).",
    )
    argparser.add_argument("--cache-size", type=int, default=0)
    args = argparser.parse_args()

    main(args)
