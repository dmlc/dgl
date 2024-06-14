"""
[For internal use only]

Demonstrate and profile the performance of sampling for link prediction tasks.
"""

import argparse
import time

import dgl

import numpy as np
import torch as th


def run(args, g, train_eids):
    fanouts = [int(fanout) for fanout in args.fanout.split(",")]

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(3)

    prob = args.prob_or_mask
    sampler = dgl.dataloading.MultiLayerNeighborSampler(
        fanouts,
        prob=prob,
    )

    exclude = None
    reverse_etypes = None
    if args.remove_edge:
        exclude = "reverse_types"
        # add reverse edge types mapping.
        reverse_etypes = {
            ("author", "affiliated_with", "institution"): (
                "institution",
                "rev-affiliated_with",
                "author",
            ),
            ("author", "writes", "paper"): ("paper", "rev-writes", "author"),
            ("paper", "has_topic", "field_of_study"): (
                "field_of_study",
                "rev-has_topic",
                "paper",
            ),
            ("paper", "cites", "paper"): ("paper", "rev-cites", "paper"),
            ("institution", "rev-affiliated_with", "author"): (
                "author",
                "affiliated_with",
                "institution",
            ),
            ("paper", "rev-writes", "author"): ("author", "writes", "paper"),
            ("field_of_study", "rev-has_topic", "paper"): (
                "paper",
                "has_topic",
                "field_of_study",
            ),
            ("paper", "rev-cites", "paper"): ("paper", "cites", "paper"),
        }

    dataloader = dgl.dataloading.DistEdgeDataLoader(
        g,
        train_eids,
        sampler,
        negative_sampler=neg_sampler,
        exclude=exclude,
        reverse_etypes=reverse_etypes,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    for epoch in range(args.n_epochs):
        sample_times = []
        tic = time.time()
        epoch_tic = time.time()
        for step, sample_data in enumerate(dataloader):
            input_nodes, pos_graph, neg_graph, blocks = sample_data

            if args.debug:
                # Verify prob/mask values.
                for block in blocks:
                    for c_etype in block.canonical_etypes:
                        homo_eids = block.edges[c_etype].data[dgl.EID]
                        assert th.all(
                            g.edges[c_etype].data[prob][homo_eids] > 0
                        )
                # Verify exclude_edges functionality.
                current_eids = blocks[-1].edata[dgl.EID]
                seed_eids = pos_graph.edata[dgl.EID]
                if exclude is None:
                    assert th.any(th.isin(current_eids, seed_eids))
                elif exclude == "self":
                    assert not th.any(th.isin(current_eids, seed_eids))
                elif exclude == "reverse_id":
                    assert not th.any(th.isin(current_eids, seed_eids))
                elif exclude == "reverse_types":
                    for src_type, etype, dst_type in pos_graph.canonical_etypes:
                        reverse_etype = reverse_etypes[
                            (src_type, etype, dst_type)
                        ]
                        seed_eids = pos_graph.edges[etype].data[dgl.EID]
                        if (src_type, etype, dst_type) in blocks[
                            -1
                        ].canonical_etypes:
                            assert not th.any(
                                th.isin(
                                    blocks[-1].edges[etype].data[dgl.EID],
                                    seed_eids,
                                )
                            )
                        if reverse_etype in blocks[-1].canonical_etypes:
                            assert not th.any(
                                th.isin(
                                    blocks[-1]
                                    .edges[reverse_etype]
                                    .data[dgl.EID],
                                    seed_eids,
                                )
                            )
                else:
                    raise ValueError(f"Unsupported exclude type: {exclude}")
            sample_times.append(time.time() - tic)
            if step % 10 == 0:
                print(
                    f"[{g.rank()}]Epoch {epoch} | Step {step} | Sample Time {np.mean(sample_times[10:]):.4f}"
                )
            tic = time.time()
        print(
            f"[{g.rank()}]Epoch {epoch} | Total time {time.time() - epoch_tic} | Sample Time {np.mean(sample_times[100:]):.4f}"
        )
        g.barrier()


def rand_init_prob(shape, dtype):
    prob = th.rand(shape)
    prob[th.randperm(len(prob))[: int(len(prob) * 0.5)]] = 0.0
    return prob


def rand_init_mask(shape, dtype):
    prob = th.rand(shape)
    prob[th.randperm(len(prob))[: int(len(prob) * 0.5)]] = 0.0
    return (prob > 0.2).to(th.float32)


def main(args):
    dgl.distributed.initialize(args.ip_config, use_graphbolt=args.use_graphbolt)

    backend = "gloo" if args.num_gpus == -1 else "nccl"
    th.distributed.init_process_group(backend=backend)

    g = dgl.distributed.DistGraph(args.graph_name)
    print("rank:", g.rank())

    # Assign prob/masks to edges.
    for c_etype in g.canonical_etypes:
        shape = (g.num_edges(etype=c_etype),)
        g.edges[c_etype].data["prob"] = dgl.distributed.DistTensor(
            shape,
            th.float32,
            init_func=rand_init_prob,
            part_policy=g.get_edge_partition_policy(c_etype),
        )
        g.edges[c_etype].data["mask"] = dgl.distributed.DistTensor(
            shape,
            th.float32,
            init_func=rand_init_mask,
            part_policy=g.get_edge_partition_policy(c_etype),
        )

    pb = g.get_partition_book()
    c_etype = ("author", "writes", "paper")
    train_eids = dgl.distributed.edge_split(
        th.ones((g.num_edges(etype=c_etype),), dtype=th.bool),
        g.get_partition_book(),
        etype=c_etype,
        force_even=True,
    )
    train_eids = {c_etype: train_eids}
    local_eids = pb.partid2eids(pb.partid, c_etype).detach().numpy()
    print(
        "part {}, train: {} (local: {})".format(
            g.rank(),
            len(train_eids[c_etype]),
            len(np.intersect1d(train_eids[c_etype].numpy(), local_eids)),
        )
    )

    run(
        args,
        g,
        train_eids,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sampling Performance Profiling For Link Prediction Tasks"
    )
    parser.add_argument("--graph-name", type=str, help="graph name")
    parser.add_argument(
        "--ip-config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument(
        "-e",
        "--n-epochs",
        type=int,
        default=5,
        help="number of training epochs",
    )
    parser.add_argument(
        "--fanout",
        type=str,
        default="4, 4",
        help="Fan-out of neighbor sampling.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=100, help="Mini-batch size. "
    )
    parser.add_argument(
        "--use_graphbolt",
        default=False,
        action="store_true",
        help="Use GraphBolt for distributed train.",
    )
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
        help="whether to remove edges during sampling",
    )
    parser.add_argument(
        "--prob_or_mask",
        type=str,
        default="prob",
        help="whether to use prob or mask during sampling",
    )
    args = parser.parse_args()

    print(args)
    main(args)
