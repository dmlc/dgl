import dgl
import dgl.function as fn
import numpy as np
import torch
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def get_ogb_evaluator(dataset):
    """
    Get evaluator from Open Graph Benchmark based on dataset
    """
    evaluator = Evaluator(name=dataset)
    return lambda preds, labels: evaluator.eval(
        {
            "y_true": labels.view(-1, 1),
            "y_pred": preds.view(-1, 1),
        }
    )["acc"]


def convert_mag_to_homograph(g, device):
    """
    Featurize node types that don't have input features (i.e. author,
    institution, field_of_study) by averaging their neighbor features.
    Then convert the graph to a undirected homogeneous graph.
    """
    src_writes, dst_writes = g.all_edges(etype="writes")
    src_topic, dst_topic = g.all_edges(etype="has_topic")
    src_aff, dst_aff = g.all_edges(etype="affiliated_with")
    new_g = dgl.heterograph(
        {
            ("paper", "written", "author"): (dst_writes, src_writes),
            ("paper", "has_topic", "field"): (src_topic, dst_topic),
            ("author", "aff", "inst"): (src_aff, dst_aff),
        }
    )
    new_g = new_g.to(device)
    new_g.nodes["paper"].data["feat"] = g.nodes["paper"].data["feat"]
    new_g["written"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["has_topic"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    new_g["aff"].update_all(fn.copy_u("feat", "m"), fn.mean("m", "feat"))
    g.nodes["author"].data["feat"] = new_g.nodes["author"].data["feat"]
    g.nodes["institution"].data["feat"] = new_g.nodes["inst"].data["feat"]
    g.nodes["field_of_study"].data["feat"] = new_g.nodes["field"].data["feat"]

    # Convert to homogeneous graph
    # Get DGL type id for paper type
    target_type_id = g.get_ntype_id("paper")
    g = dgl.to_homogeneous(g, ndata=["feat"])
    g = dgl.add_reverse_edges(g, copy_ndata=True)
    # Mask for paper nodes
    g.ndata["target_mask"] = g.ndata[dgl.NTYPE] == target_type_id
    return g


def load_dataset(name, device):
    """
    Load dataset and move graph and features to device
    """
    if name not in ["ogbn-products", "ogbn-arxiv", "ogbn-mag"]:
        raise RuntimeError("Dataset {} is not supported".format(name))
    dataset = DglNodePropPredDataset(name=name)
    splitted_idx = dataset.get_idx_split()
    train_nid = splitted_idx["train"]
    val_nid = splitted_idx["valid"]
    test_nid = splitted_idx["test"]
    g, labels = dataset[0]
    g = g.to(device)
    if name == "ogbn-arxiv":
        g = dgl.add_reverse_edges(g, copy_ndata=True)
        g = dgl.add_self_loop(g)
        g.ndata["feat"] = g.ndata["feat"].float()
    elif name == "ogbn-mag":
        # MAG is a heterogeneous graph. The task is to make prediction for
        # paper nodes
        labels = labels["paper"]
        train_nid = train_nid["paper"]
        val_nid = val_nid["paper"]
        test_nid = test_nid["paper"]
        g = convert_mag_to_homograph(g, device)
    else:
        g.ndata["feat"] = g.ndata["feat"].float()
    n_classes = dataset.num_classes
    labels = labels.squeeze()
    evaluator = get_ogb_evaluator(name)

    print(
        f"# Nodes: {g.num_nodes()}\n"
        f"# Edges: {g.num_edges()}\n"
        f"# Train: {len(train_nid)}\n"
        f"# Val: {len(val_nid)}\n"
        f"# Test: {len(test_nid)}\n"
        f"# Classes: {n_classes}"
    )

    return g, labels, n_classes, train_nid, val_nid, test_nid, evaluator
