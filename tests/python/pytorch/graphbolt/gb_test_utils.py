import os

import dgl
import dgl.graphbolt as gb

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def rand_csc_graph(N, density, bidirection_edge=False):
    adj = sp.random(N, N, density)
    if bidirection_edge:
        adj = adj + adj.T
    adj = adj.tocsc()

    indptr = torch.LongTensor(adj.indptr)
    indices = torch.LongTensor(adj.indices)

    graph = gb.fused_csc_sampling_graph(indptr, indices)

    return graph


def random_homo_graph(num_nodes, num_edges):
    csc_indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    csc_indptr = torch.sort(csc_indptr)[0]
    csc_indptr[0] = 0
    csc_indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return csc_indptr, indices


def get_type_to_id(num_ntypes, num_etypes):
    ntypes = {f"n{i}": i for i in range(num_ntypes)}
    etypes = {}
    count = 0
    for n1 in range(num_ntypes):
        for n2 in range(n1, num_ntypes):
            if count >= num_etypes:
                break
            etypes.update({f"n{n1}:e{count}:n{n2}": count})
            count += 1
    return ntypes, etypes


def get_ntypes_and_etypes(num_nodes, num_ntypes, num_etypes):
    ntypes = {f"n{i}": num_nodes // num_ntypes for i in range(num_ntypes)}
    if num_nodes % num_ntypes != 0:
        ntypes["n0"] += num_nodes % num_ntypes
    etypes = []
    count = 0
    while count < num_etypes:
        for n1 in range(num_ntypes):
            for n2 in range(num_ntypes):
                if count >= num_etypes:
                    break
                etypes.append((f"n{n1}", f"e{count}", f"n{n2}"))
                count += 1
    return ntypes, etypes


def random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    ntypes, etypes = get_ntypes_and_etypes(num_nodes, num_ntypes, num_etypes)
    edges = {}
    for step, etype in enumerate(etypes):
        src_ntype, _, dst_ntype = etype
        num_e = num_edges // num_etypes + (
            0 if step != 0 else num_edges % num_etypes
        )
        if ntypes[src_ntype] == 0 or ntypes[dst_ntype] == 0:
            continue
        src = torch.randint(0, ntypes[src_ntype], (num_e,))
        dst = torch.randint(0, ntypes[dst_ntype], (num_e,))

        edges[etype] = (src, dst)

    gb_g = gb.from_dglgraph(dgl.heterograph(edges, ntypes))
    return (
        gb_g.csc_indptr,
        gb_g.indices,
        gb_g.node_type_offset,
        gb_g.type_per_edge,
        gb_g.node_type_to_id,
        gb_g.edge_type_to_id,
    )


def random_homo_graphbolt_graph(
    test_dir, dataset_name, num_nodes, num_edges, num_classes, edge_fmt="csv"
):
    """Generate random graphbolt version homograph"""
    # Generate random edges.
    nodes = np.repeat(np.arange(num_nodes, dtype=np.int64), 5)
    neighbors = np.random.randint(
        0, num_nodes, size=(num_edges), dtype=np.int64
    )
    edges = np.stack([nodes, neighbors], axis=1)
    os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
    assert edge_fmt in [
        "numpy",
        "csv",
    ], "Only numpy and csv are supported for edges."
    if edge_fmt == "csv":
        # Write into edges/edge.csv
        edges_DataFrame = pd.DataFrame(edges, columns=["src", "dst"])
        edge_path = os.path.join("edges", "edge.csv")
        edges_DataFrame.to_csv(
            os.path.join(test_dir, edge_path),
            index=False,
            header=False,
        )
    else:
        # Write into edges/edge.npy
        edges = edges.T
        edge_path = os.path.join("edges", "edge.npy")
        np.save(os.path.join(test_dir, edge_path), edges)

    # Generate random graph edge-feats.
    edge_feats = np.random.rand(num_edges, num_classes)
    os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
    edge_feat_path = os.path.join("data", "edge-feat.npy")
    np.save(os.path.join(test_dir, edge_feat_path), edge_feats)

    # Generate random node-feats.
    if num_classes == 1:
        node_feats = np.random.rand(num_nodes)
    else:
        node_feats = np.random.rand(num_nodes, num_classes)
    node_feat_path = os.path.join("data", "node-feat.npy")
    np.save(os.path.join(test_dir, node_feat_path), node_feats)

    # Generate train/test/valid set.
    assert num_nodes % 4 == 0, "num_nodes must be divisible by 4"
    each_set_size = num_nodes // 4
    os.makedirs(os.path.join(test_dir, "set"), exist_ok=True)
    train_pairs = (
        np.arange(each_set_size),
        np.arange(each_set_size, 2 * each_set_size),
    )
    train_data = np.vstack(train_pairs).T.astype(edges.dtype)
    train_path = os.path.join("set", "train.npy")
    np.save(os.path.join(test_dir, train_path), train_data)

    validation_pairs = (
        np.arange(each_set_size, 2 * each_set_size),
        np.arange(2 * each_set_size, 3 * each_set_size),
    )
    validation_data = np.vstack(validation_pairs).T.astype(edges.dtype)
    validation_path = os.path.join("set", "validation.npy")
    np.save(os.path.join(test_dir, validation_path), validation_data)

    test_pairs = (
        np.arange(2 * each_set_size, 3 * each_set_size),
        np.arange(3 * each_set_size, 4 * each_set_size),
    )
    test_data = np.vstack(test_pairs).T.astype(edges.dtype)
    test_path = os.path.join("set", "test.npy")
    np.save(os.path.join(test_dir, test_path), test_data)

    yaml_content = f"""
        dataset_name: {dataset_name}
        graph: # Graph structure and required attributes.
            nodes:
                - num: {num_nodes}
            edges:
                - format: {edge_fmt}
                  path: {edge_path}
            feature_data:
                - domain: node
                  type: null
                  name: feat
                  format: numpy
                  in_memory: true
                  path: {node_feat_path}
                - domain: edge
                  type: null
                  name: feat
                  format: numpy
                  in_memory: true
                  path: {edge_feat_path}
        feature_data:
            - domain: node
              type: null
              name: feat
              format: numpy
              in_memory: true
              path: {node_feat_path}
            - domain: edge
              type: null
              name: feat
              format: numpy
              path: {edge_feat_path}
        tasks:
          - name: link_prediction
            num_classes: {num_classes}
            train_set:
              - type: null
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {train_path}
            validation_set:
              - type: null
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {validation_path}
            test_set:
              - type: null
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {test_path}
    """
    return yaml_content


def generate_raw_data_for_hetero_dataset(
    test_dir, dataset_name, num_nodes, num_edges, num_classes, edge_fmt="csv"
):
    # Generate edges.
    edges_path = {}
    for etype, num_edge in num_edges.items():
        src_ntype, etype_str, dst_ntype = etype
        src = torch.randint(0, num_nodes[src_ntype], (num_edge,))
        dst = torch.randint(0, num_nodes[dst_ntype], (num_edge,))
        os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
        assert edge_fmt in [
            "numpy",
            "csv",
        ], "Only numpy and csv are supported for edges."
        if edge_fmt == "csv":
            # Write into edges/edge.csv
            edges = pd.DataFrame(
                np.stack([src, dst], axis=1), columns=["src", "dst"]
            )
            edge_path = os.path.join("edges", f"{etype_str}.csv")
            edges.to_csv(
                os.path.join(test_dir, edge_path),
                index=False,
                header=False,
            )
        else:
            edges = np.stack([src, dst], axis=1).T
            edge_path = os.path.join("edges", f"{etype_str}.npy")
            np.save(os.path.join(test_dir, edge_path), edges)
        edges_path[etype_str] = edge_path

    # Generate node features.
    node_feats_path = {}
    os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
    for ntype, num_node in num_nodes.items():
        node_feat_path = os.path.join("data", f"{ntype}-feat.npy")
        node_feats = np.random.rand(num_node, num_classes)
        np.save(os.path.join(test_dir, node_feat_path), node_feats)
        node_feats_path[ntype] = node_feat_path

    # Generate edge features.
    edge_feats_path = {}
    os.makedirs(os.path.join(test_dir, "data"), exist_ok=True)
    for etype, num_edge in num_edges.items():
        src_ntype, etype_str, dst_ntype = etype
        edge_feat_path = os.path.join("data", f"{etype_str}-feat.npy")
        edge_feats = np.random.rand(num_edge, num_classes)
        np.save(os.path.join(test_dir, edge_feat_path), edge_feats)
        edge_feats_path[etype_str] = edge_feat_path

    # Generate train/test/valid set.
    os.makedirs(os.path.join(test_dir, "set"), exist_ok=True)
    user_ids = torch.arange(num_nodes["user"])
    np.random.shuffle(user_ids.numpy())
    num_train = int(num_nodes["user"] * 0.6)
    num_validation = int(num_nodes["user"] * 0.2)
    num_test = num_nodes["user"] - num_train - num_validation
    train_path = os.path.join("set", "train.npy")
    np.save(os.path.join(test_dir, train_path), user_ids[:num_train])
    validation_path = os.path.join("set", "validation.npy")
    np.save(
        os.path.join(test_dir, validation_path),
        user_ids[num_train : num_train + num_validation],
    )
    test_path = os.path.join("set", "test.npy")
    np.save(
        os.path.join(test_dir, test_path),
        user_ids[num_train + num_validation :],
    )

    yaml_content = f"""
        dataset_name: {dataset_name}
        graph: # Graph structure and required attributes.
          nodes:
            - type: user
              num: {num_nodes["user"]}
            - type: item
              num: {num_nodes["item"]}
          edges:
            - type: "user:follow:user"
              format: {edge_fmt}
              path: {edges_path["follow"]}
            - type: "user:click:item"
              format: {edge_fmt}
              path: {edges_path["click"]}
          feature_data:
            - domain: node
              type: user
              name: feat
              format: numpy
              in_memory: true
              path: {node_feats_path["user"]}
            - domain: node
              type: item
              name: feat
              format: numpy
              in_memory: true
              path: {node_feats_path["item"]}
            - domain: edge
              type: "user:follow:user"
              name: feat
              format: numpy
              in_memory: true
              path: {edge_feats_path["follow"]}
            - domain: edge
              type: "user:click:item"
              name: feat
              format: numpy
              in_memory: true
              path: {edge_feats_path["click"]}
        feature_data:
          - domain: node
            type: user
            name: feat
            format: numpy
            in_memory: true
            path: {node_feats_path["user"]}
          - domain: node
            type: item
            name: feat
            format: numpy
            in_memory: true
            path: {node_feats_path["item"]}
        tasks:
          - name: node_classification
            num_classes: {num_classes}
            train_set:
              - type: user
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {train_path}
            validation_set:
              - type: user
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {validation_path}
            test_set:
              - type: user
                data:
                  - name: seeds
                    format: numpy
                    in_memory: true
                    path: {test_path}
    """

    yaml_file = os.path.join(test_dir, "metadata.yaml")
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
