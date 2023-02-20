""" load dataset from ogb """

import argparse

import dgl

from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset


def load_from_ogbl_with_name(name):
    choices = ["ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation"]
    assert name in choices, "name must be selected from " + str(choices)
    dataset = DglLinkPropPredDataset(name)
    return dataset[0]


def load_from_ogbn_with_name(name):
    choices = [
        "ogbn-products",
        "ogbn-proteins",
        "ogbn-arxiv",
        "ogbn-papers100M",
    ]
    assert name in choices, "name must be selected from " + str(choices)
    dataset, label = DglNodePropPredDataset(name)[0]
    return dataset


if __name__ == "__main__":
    """load datasets as net.txt format"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        type=str,
        choices=[
            "ogbl-collab",
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-citation",
            "ogbn-products",
            "ogbn-proteins",
            "ogbn-arxiv",
            "ogbn-papers100M",
        ],
        default="ogbl-collab",
        help="name of datasets by ogb",
    )
    args = parser.parse_args()

    name = args.name
    if name.startswith("ogbl"):
        g = load_from_ogbl_with_name(name=name)
    else:
        g = load_from_ogbn_with_name(name=name)

    dgl.save_graphs(name + "-graph.bin", g)
