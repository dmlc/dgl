import argparse
import os

import dgl

import torch as th
from dgl import load_graphs
from dgl.data import (
    BACommunityDataset,
    BAShapeDataset,
    TreeCycleDataset,
    TreeGridDataset,
)
from dgl.nn import GNNExplainer
from gnnlens import Writer
from models import Model


def main(args):
    if args.dataset == "BAShape":
        dataset = BAShapeDataset(seed=0)
    elif args.dataset == "BACommunity":
        dataset = BACommunityDataset(seed=0)
    elif args.dataset == "TreeCycle":
        dataset = TreeCycleDataset(seed=0)
    elif args.dataset == "TreeGrid":
        dataset = TreeGridDataset(seed=0)

    graph = dataset[0]
    labels = graph.ndata["label"]
    feats = graph.ndata["feat"]
    num_classes = dataset.num_classes

    # load an existing model
    model_path = os.path.join("./", f"model_{args.dataset}.pth")
    model_stat_dict = th.load(model_path)
    model = Model(feats.shape[-1], num_classes)
    model.load_state_dict(model_stat_dict)

    # Choose the first node of the class 1 for explaining prediction
    target_class = 1
    for n_idx, n_label in enumerate(labels):
        if n_label == target_class:
            break

    explainer = GNNExplainer(model, num_hops=3)
    new_center, sub_graph, feat_mask, edge_mask = explainer.explain_node(
        n_idx, graph, feats
    )

    # gnnlens2
    # Specify the path to create a new directory for dumping data files.
    writer = Writer("gnn_subgraph")
    writer.add_graph(
        name=args.dataset,
        graph=graph,
        nlabels=labels,
        num_nlabel_types=num_classes,
    )
    writer.add_subgraph(
        graph_name=args.dataset,
        subgraph_name="GNNExplainer",
        node_id=n_idx,
        subgraph_nids=sub_graph.ndata[dgl.NID],
        subgraph_eids=sub_graph.edata[dgl.EID],
        subgraph_eweights=edge_mask,
    )

    # Finish dumping.
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Demo of GNN explainer in DGL")
    parser.add_argument(
        "--dataset",
        type=str,
        default="BAShape",
        choices=["BAShape", "BACommunity", "TreeCycle", "TreeGrid"],
    )
    args = parser.parse_args()
    print(args)

    main(args)
