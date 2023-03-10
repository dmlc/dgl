import argparse
import multiprocessing
import time
from collections import defaultdict
from functools import partial, reduce, wraps

import networkx as nx
import numpy as np
import torch
from gensim.models.keyedvectors import Vocab
from six import iteritems
from sklearn.metrics import auc, f1_score, precision_recall_curve, roc_auc_score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input", type=str, default="data/amazon", help="Input dataset path"
    )

    parser.add_argument(
        "--features", type=str, default=None, help="Input node features"
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="Number of epoch. Default is 100.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of batch_size. Default is 64.",
    )

    parser.add_argument(
        "--eval-type",
        type=str,
        default="all",
        help="The edge type(s) for evaluation.",
    )

    parser.add_argument(
        "--schema",
        type=str,
        default=None,
        help="The metapath schema (e.g., U-I-U,I-U-I).",
    )

    parser.add_argument(
        "--dimensions",
        type=int,
        default=200,
        help="Number of dimensions. Default is 200.",
    )

    parser.add_argument(
        "--edge-dim",
        type=int,
        default=10,
        help="Number of edge embedding dimensions. Default is 10.",
    )

    parser.add_argument(
        "--att-dim",
        type=int,
        default=20,
        help="Number of attention dimensions. Default is 20.",
    )

    parser.add_argument(
        "--walk-length",
        type=int,
        default=10,
        help="Length of walk per source. Default is 10.",
    )

    parser.add_argument(
        "--num-walks",
        type=int,
        default=20,
        help="Number of walks per source. Default is 20.",
    )

    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Context size for optimization. Default is 5.",
    )

    parser.add_argument(
        "--negative-samples",
        type=int,
        default=5,
        help="Negative samples for optimization. Default is 5.",
    )

    parser.add_argument(
        "--neighbor-samples",
        type=int,
        default=10,
        help="Neighbor samples for aggregation. Default is 10.",
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience. Default is 5.",
    )

    parser.add_argument(
        "--gpu",
        type=str,
        default=None,
        help="Comma separated list of GPU device IDs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers.",
    )

    return parser.parse_args()


# for each line, the data is [edge_type, node, node]
def load_training_data(f_name):
    print("We are loading data from:", f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, "r") as f:
        for line in f:
            words = line[:-1].split(" ")  # line[-1] == '\n'
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print("Total training nodes: " + str(len(all_nodes)))
    return edge_data_by_type


# for each line, the data is [edge_type, node, node, true_or_false]
def load_testing_data(f_name):
    print("We are loading data from:", f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_edges = list()
    all_nodes = list()
    with open(f_name, "r") as f:
        for line in f:
            words = line[:-1].split(" ")
            x, y = words[1], words[2]
            if int(words[3]) == 1:
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(f_name):
    print("We are loading node type from:", f_name)
    node_type = {}
    with open(f_name, "r") as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type


def generate_pairs_parallel(walks, skip_window=None, layer_id=None):
    pairs = []
    for walk in walks:
        walk = walk.tolist()
        for i in range(len(walk)):
            for j in range(1, skip_window + 1):
                if i - j >= 0:
                    pairs.append((walk[i], walk[i - j], layer_id))
                if i + j < len(walk):
                    pairs.append((walk[i], walk[i + j], layer_id))
    return pairs


def generate_pairs(all_walks, window_size, num_workers):
    # for each node, choose the first neighbor and second neighbor of it to form pairs
    # Get all worker processes
    start_time = time.time()
    print("We are generating pairs with {} cores.".format(num_workers))

    # Start all worker processes
    pool = multiprocessing.Pool(processes=num_workers)
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        block_num = len(walks) // num_workers
        if block_num > 0:
            walks_list = [
                walks[i * block_num : min((i + 1) * block_num, len(walks))]
                for i in range(num_workers)
            ]
        else:
            walks_list = [walks]
        tmp_result = pool.map(
            partial(
                generate_pairs_parallel,
                skip_window=skip_window,
                layer_id=layer_id,
            ),
            walks_list,
        )
        pairs += reduce(lambda x, y: x + y, tmp_result)

    pool.close()
    end_time = time.time()
    print("Generate pairs end, use {}s.".format(end_time - start_time))
    return np.array([list(pair) for pair in set(pairs)])


def generate_vocab(network_data):
    nodes, index2word = [], []
    for edge_type in network_data:
        node1, node2 = zip(*network_data[edge_type])
        index2word = index2word + list(node1) + list(node2)

    index2word = list(set(index2word))
    vocab = {}
    i = 0
    for word in index2word:
        vocab[word] = i
        i = i + 1

    for edge_type in network_data:
        node1, node2 = zip(*network_data[edge_type])
        tmp_nodes = list(set(list(node1) + list(node2)))
        tmp_nodes = [vocab[word] for word in tmp_nodes]
        nodes.append(tmp_nodes)

    return index2word, vocab, nodes


def get_score(local_model, edge):
    node1, node2 = str(edge[0]), str(edge[1])
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges, num_workers):
    true_list = list()
    prediction_list = list()
    true_num = 0

    # Start all worker processes
    pool = multiprocessing.Pool(processes=num_workers)
    tmp_true_score_list = pool.map(partial(get_score, model), true_edges)
    tmp_false_score_list = pool.map(partial(get_score, model), false_edges)
    pool.close()

    prediction_list += [
        tmp_score for tmp_score in tmp_true_score_list if tmp_score is not None
    ]
    true_num = len(prediction_list)
    true_list += [1] * true_num

    prediction_list += [
        tmp_score for tmp_score in tmp_false_score_list if tmp_score is not None
    ]
    true_list += [0] * (len(prediction_list) - true_num)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)
    return (
        roc_auc_score(y_true, y_scores),
        f1_score(y_true, y_pred),
        auc(rs, ps),
    )
