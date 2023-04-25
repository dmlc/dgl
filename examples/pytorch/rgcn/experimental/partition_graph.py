import argparse
import time

import dgl
import numpy as np
import torch as th

from ogb.nodeproppred import DglNodePropPredDataset


def load_ogb(dataset):
    if dataset == "ogbn-mag":
        dataset = DglNodePropPredDataset(name=dataset)
        split_idx = dataset.get_idx_split()
        train_idx = split_idx["train"]["paper"]
        val_idx = split_idx["valid"]["paper"]
        test_idx = split_idx["test"]["paper"]
        hg_orig, labels = dataset[0]
        subgs = {}
        for etype in hg_orig.canonical_etypes:
            u, v = hg_orig.all_edges(etype=etype)
            subgs[etype] = (u, v)
            subgs[(etype[2], "rev-" + etype[1], etype[0])] = (v, u)
        hg = dgl.heterograph(subgs)
        hg.nodes["paper"].data["feat"] = hg_orig.nodes["paper"].data["feat"]
        paper_labels = labels["paper"].squeeze()

        num_rels = len(hg.canonical_etypes)
        num_of_ntype = len(hg.ntypes)
        num_classes = dataset.num_classes
        category = "paper"
        print("Number of relations: {}".format(num_rels))
        print("Number of class: {}".format(num_classes))
        print("Number of train: {}".format(len(train_idx)))
        print("Number of valid: {}".format(len(val_idx)))
        print("Number of test: {}".format(len(test_idx)))

        # get target category id
        category_id = len(hg.ntypes)
        for i, ntype in enumerate(hg.ntypes):
            if ntype == category:
                category_id = i

        train_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        train_mask[train_idx] = True
        val_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        val_mask[val_idx] = True
        test_mask = th.zeros((hg.num_nodes("paper"),), dtype=th.bool)
        test_mask[test_idx] = True
        hg.nodes["paper"].data["train_mask"] = train_mask
        hg.nodes["paper"].data["val_mask"] = val_mask
        hg.nodes["paper"].data["test_mask"] = test_mask

        hg.nodes["paper"].data["labels"] = paper_labels
        return hg
    else:
        raise ("Do not support other ogbn datasets.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument(
        "--dataset", type=str, default="ogbn-mag", help="datasets: ogbn-mag"
    )
    argparser.add_argument(
        "--num_parts", type=int, default=4, help="number of partitions"
    )
    argparser.add_argument(
        "--part_method", type=str, default="metis", help="the partition method"
    )
    argparser.add_argument(
        "--balance_train",
        action="store_true",
        help="balance the training size in each partition.",
    )
    argparser.add_argument(
        "--undirected",
        action="store_true",
        help="turn the graph into an undirected graph.",
    )
    argparser.add_argument(
        "--balance_edges",
        action="store_true",
        help="balance the number of edges in each partition.",
    )
    argparser.add_argument(
        "--num_trainers_per_machine",
        type=int,
        default=1,
        help="the number of trainers per machine. The trainer ids are stored\
                                in the node feature 'trainer_id'",
    )
    argparser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output path of partitioned graph.",
    )
    args = argparser.parse_args()

    start = time.time()
    g = load_ogb(args.dataset)

    print(
        "load {} takes {:.3f} seconds".format(args.dataset, time.time() - start)
    )
    print("|V|={}, |E|={}".format(g.num_nodes(), g.num_edges()))
    print(
        "train: {}, valid: {}, test: {}".format(
            th.sum(g.nodes["paper"].data["train_mask"]),
            th.sum(g.nodes["paper"].data["val_mask"]),
            th.sum(g.nodes["paper"].data["test_mask"]),
        )
    )

    if args.balance_train:
        balance_ntypes = {"paper": g.nodes["paper"].data["train_mask"]}
    else:
        balance_ntypes = None

    dgl.distributed.partition_graph(
        g,
        args.dataset,
        args.num_parts,
        args.output,
        part_method=args.part_method,
        balance_ntypes=balance_ntypes,
        balance_edges=args.balance_edges,
        num_trainers_per_machine=args.num_trainers_per_machine,
    )
