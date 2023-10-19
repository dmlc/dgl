import os

import dgl.graphbolt as gb

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def rand_csc_graph(N, density):
    adj = sp.random(N, N, density)
    adj = adj + adj.T
    adj = adj.tocsc()

    indptr = torch.LongTensor(adj.indptr)
    indices = torch.LongTensor(adj.indices)

    graph = gb.from_csc(indptr, indices)

    return graph


def random_homo_graph(num_nodes, num_edges):
    csc_indptr = torch.randint(0, num_edges, (num_nodes + 1,))
    csc_indptr = torch.sort(csc_indptr)[0]
    csc_indptr[0] = 0
    csc_indptr[-1] = num_edges
    indices = torch.randint(0, num_nodes, (num_edges,))
    return csc_indptr, indices


def get_metadata(num_ntypes, num_etypes):
    ntypes = {f"n{i}": i for i in range(num_ntypes)}
    etypes = {}
    count = 0
    for n1 in range(num_ntypes):
        for n2 in range(n1, num_ntypes):
            if count >= num_etypes:
                break
            etypes.update({f"n{n1}:e{count}:n{n2}": count})
            count += 1
    return gb.GraphMetadata(ntypes, etypes)


def random_hetero_graph(num_nodes, num_edges, num_ntypes, num_etypes):
    csc_indptr, indices = random_homo_graph(num_nodes, num_edges)
    metadata = get_metadata(num_ntypes, num_etypes)
    # Randomly get node type split point.
    node_type_offset = torch.sort(
        torch.randint(0, num_nodes, (num_ntypes + 1,))
    )[0]
    node_type_offset[0] = 0
    node_type_offset[-1] = num_nodes

    type_per_edge = []
    for i in range(num_nodes):
        num = csc_indptr[i + 1] - csc_indptr[i]
        type_per_edge.append(
            torch.sort(torch.randint(0, num_etypes, (num,)))[0]
        )
    type_per_edge = torch.cat(type_per_edge, dim=0)
    return (csc_indptr, indices, node_type_offset, type_per_edge, metadata)


def random_homo_graphbolt_graph(
    test_dir, dataset_name, num_nodes, num_edges, num_classes
):
    """Generate random graphbolt version homograph"""
    # Generate random edges.
    nodes = np.repeat(np.arange(num_nodes), 5)
    neighbors = np.random.randint(0, num_nodes, size=(num_edges))
    edges = np.stack([nodes, neighbors], axis=1)
    # Wrtie into edges/edge.csv
    os.makedirs(os.path.join(test_dir, "edges"), exist_ok=True)
    edges = pd.DataFrame(edges, columns=["src", "dst"])
    edge_path = os.path.join("edges", "edge.csv")
    edges.to_csv(
        os.path.join(test_dir, edge_path),
        index=False,
        header=False,
    )

    # Generate random graph edge-feats.
    edge_feats = np.random.rand(num_edges, 5)
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
    train_labels = np.random.randint(0, num_classes, size=each_set_size)
    train_data = np.vstack([train_pairs, train_labels]).T
    train_path = os.path.join("set", "train.npy")
    np.save(os.path.join(test_dir, train_path), train_data)

    validation_pairs = (
        np.arange(each_set_size, 2 * each_set_size),
        np.arange(2 * each_set_size, 3 * each_set_size),
    )
    validation_labels = np.random.randint(0, num_classes, size=each_set_size)
    validation_data = np.vstack([validation_pairs, validation_labels]).T
    validation_path = os.path.join("set", "validation.npy")
    np.save(os.path.join(test_dir, validation_path), validation_data)

    test_pairs = (
        np.arange(2 * each_set_size, 3 * each_set_size),
        np.arange(3 * each_set_size, 4 * each_set_size),
    )
    test_labels = np.random.randint(0, num_classes, size=each_set_size)
    test_data = np.vstack([test_pairs, test_labels]).T
    test_path = os.path.join("set", "test.npy")
    np.save(os.path.join(test_dir, test_path), test_data)

    yaml_content = f"""
        dataset_name: {dataset_name}
        graph: # graph structure and required attributes.
            nodes:
                - num: {num_nodes}
            edges:
                - format: csv
                  path: {edge_path}
            feature_data:
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
              in_memory: false
              path: {node_feat_path}
        tasks:
          - name: node_classification
            num_classes: {num_classes}
            train_set:
              - type_name: null
                data:
                  - format: numpy
                    path: {train_path}
            validation_set:
              - type_name: null
                data:
                  - format: numpy
                    path: {validation_path}
            test_set:
              - type_name: null
                data:
                  - format: numpy
                    path: {test_path}
    """
    return yaml_content
