import dgl
import torch as th

from dgl.data.rdf import AIFBDataset, AMDataset, BGSDataset, MUTAGDataset


def load_data(data_name, get_norm=False, inv_target=False):
    if data_name == "aifb":
        dataset = AIFBDataset()
    elif data_name == "mutag":
        dataset = MUTAGDataset()
    elif data_name == "bgs":
        dataset = BGSDataset()
    else:
        dataset = AMDataset()

    # Load hetero-graph
    hg = dataset[0]

    num_rels = len(hg.canonical_etypes)
    category = dataset.predict_category
    num_classes = dataset.num_classes
    labels = hg.nodes[category].data.pop("labels")
    train_mask = hg.nodes[category].data.pop("train_mask")
    test_mask = hg.nodes[category].data.pop("test_mask")
    train_idx = th.nonzero(train_mask, as_tuple=False).squeeze()
    test_idx = th.nonzero(test_mask, as_tuple=False).squeeze()

    if get_norm:
        # Calculate normalization weight for each edge,
        # 1. / d, d is the degree of the destination node
        for cetype in hg.canonical_etypes:
            hg.edges[cetype].data["norm"] = dgl.norm_by_dst(
                hg, cetype
            ).unsqueeze(1)
        edata = ["norm"]
    else:
        edata = None

    # get target category id
    category_id = hg.ntypes.index(category)

    g = dgl.to_homogeneous(hg, edata=edata)
    # Rename the fields as they can be changed by for example DataLoader
    g.ndata["ntype"] = g.ndata.pop(dgl.NTYPE)
    g.ndata["type_id"] = g.ndata.pop(dgl.NID)
    node_ids = th.arange(g.num_nodes())

    # find out the target node ids in g
    loc = g.ndata["ntype"] == category_id
    target_idx = node_ids[loc]

    if inv_target:
        # Map global node IDs to type-specific node IDs. This is required for
        # looking up type-specific labels in a minibatch
        inv_target = th.empty((g.num_nodes(),), dtype=th.int64)
        inv_target[target_idx] = th.arange(
            0, target_idx.shape[0], dtype=inv_target.dtype
        )
        return (
            g,
            num_rels,
            num_classes,
            labels,
            train_idx,
            test_idx,
            target_idx,
            inv_target,
        )
    else:
        return g, num_rels, num_classes, labels, train_idx, test_idx, target_idx
