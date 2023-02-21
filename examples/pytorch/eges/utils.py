import argparse
import random
from datetime import datetime

import dgl

import networkx as nx
import numpy as np
import torch as th


def init_args():
    # TODO: change args
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--session_interval_sec", type=int, default=1800)
    argparser.add_argument(
        "--action_data", type=str, default="data/action_head.csv"
    )
    argparser.add_argument(
        "--item_info_data", type=str, default="data/jdata_product.csv"
    )
    argparser.add_argument("--walk_length", type=int, default=10)
    argparser.add_argument("--num_walks", type=int, default=5)
    argparser.add_argument("--batch_size", type=int, default=64)
    argparser.add_argument("--dim", type=int, default=16)
    argparser.add_argument("--epochs", type=int, default=30)
    argparser.add_argument("--window_size", type=int, default=2)
    argparser.add_argument("--num_negative", type=int, default=5)
    argparser.add_argument("--lr", type=float, default=0.001)
    argparser.add_argument("--log_every", type=int, default=100)

    return argparser.parse_args()


def construct_graph(datapath, session_interval_gap_sec, valid_sku_raw_ids):
    user_clicks, sku_encoder, sku_decoder = parse_actions(
        datapath, valid_sku_raw_ids
    )

    # {src,dst: weight}
    graph = {}
    for user_id, action_list in user_clicks.items():
        # sort by action time
        _action_list = sorted(action_list, key=lambda x: x[1])

        last_action_time = datetime.strptime(
            _action_list[0][1], "%Y-%m-%d %H:%M:%S"
        )
        session = [_action_list[0][0]]
        # cut sessions and add to graph
        for sku_id, action_time in _action_list[1:]:
            action_time = datetime.strptime(action_time, "%Y-%m-%d %H:%M:%S")
            gap = action_time - last_action_time
            if gap.seconds < session_interval_gap_sec:
                session.append(sku_id)
            else:
                # here we have a new session
                # add prev session to graph
                add_session(session, graph)
                # create a new session
                session = [sku_id]
        # add last session
        add_session(session, graph)

    g = convert_to_dgl_graph(graph)

    return g, sku_encoder, sku_decoder


def convert_to_dgl_graph(graph):
    # directed graph
    g = nx.DiGraph()
    for edge, weight in graph.items():
        nodes = edge.split(",")
        src, dst = int(nodes[0]), int(nodes[1])
        g.add_edge(src, dst, weight=float(weight))

    return dgl.from_networkx(g, edge_attrs=["weight"])


def add_session(session, graph):
    """
    For session like:
        [sku1, sku2, sku3]
    add 1 weight to each of the following edges:
        sku1 -> sku2
        sku2 -> sku3
    If sesson length < 2, no nodes/edges will be added
    """
    for i in range(len(session) - 1):
        edge = str(session[i]) + "," + str(session[i + 1])
        try:
            graph[edge] += 1
        except KeyError:
            graph[edge] = 1


def parse_actions(datapath, valid_sku_raw_ids):
    user_clicks = {}
    with open(datapath, "r") as f:
        f.readline()
        # raw_id -> new_id and new_id -> raw_id
        sku_encoder, sku_decoder = {}, []
        sku_id = -1
        for line in f:
            line = line.replace("\n", "")
            fields = line.split(",")
            action_type = fields[-1]
            # actually, all types in the dataset is "1"
            if action_type == "1":
                user_id = fields[0]
                sku_raw_id = fields[1]
                if sku_raw_id in valid_sku_raw_ids:
                    action_time = fields[2]
                    # encode sku_id
                    sku_id = encode_id(
                        sku_encoder, sku_decoder, sku_raw_id, sku_id
                    )

                    # add to user clicks
                    try:
                        user_clicks[user_id].append((sku_id, action_time))
                    except KeyError:
                        user_clicks[user_id] = [(sku_id, action_time)]

    return user_clicks, sku_encoder, sku_decoder


def encode_id(encoder, decoder, raw_id, encoded_id):
    if raw_id in encoder:
        return encoded_id
    else:
        encoded_id += 1
        encoder[raw_id] = encoded_id
        decoder.append(raw_id)

    return encoded_id


def get_valid_sku_set(datapath):
    sku_ids = set()
    with open(datapath, "r") as f:
        for line in f.readlines():
            line.replace("\n", "")
            sku_raw_id = line.split(",")[0]
            sku_ids.add(sku_raw_id)

    return sku_ids


def encode_sku_fields(datapath, sku_encoder, sku_decoder):
    # sku_id,brand,shop_id,cate,market_time
    sku_info_encoder = {"brand": {}, "shop": {}, "cate": {}}
    sku_info_decoder = {"brand": [], "shop": [], "cate": []}
    sku_info = {}
    brand_id, shop_id, cate_id = -1, -1, -1
    with open(datapath, "r") as f:
        f.readline()
        for line in f:
            line = line.replace("\n", "")
            fields = line.split(",")
            sku_raw_id = fields[0]

            brand_raw_id = fields[1]
            shop_raw_id = fields[2]
            cate_raw_id = fields[3]

            if sku_raw_id in sku_encoder:
                sku_id = sku_encoder[sku_raw_id]

                brand_id = encode_id(
                    sku_info_encoder["brand"],
                    sku_info_decoder["brand"],
                    brand_raw_id,
                    brand_id,
                )

                shop_id = encode_id(
                    sku_info_encoder["shop"],
                    sku_info_decoder["shop"],
                    shop_raw_id,
                    shop_id,
                )

                cate_id = encode_id(
                    sku_info_encoder["cate"],
                    sku_info_decoder["cate"],
                    cate_raw_id,
                    cate_id,
                )

                sku_info[sku_id] = [sku_id, brand_id, shop_id, cate_id]

    return sku_info_encoder, sku_info_decoder, sku_info


class TestEdge:
    def __init__(self, src, dst, label):
        self.src = src
        self.dst = dst
        self.label = label


def split_train_test_graph(graph):
    """
    For test true edges, 1/3 of the edges are randomly chosen
    and removed as ground truth in the test set,
    the remaining graph is taken as the training set.
    """
    test_edges = []
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    sampled_edge_ids = random.sample(
        range(graph.num_edges()), int(graph.num_edges() / 3)
    )
    for edge_id in sampled_edge_ids:
        src, dst = graph.find_edges(edge_id)
        test_edges.append(TestEdge(src, dst, 1))

        src, dst = neg_sampler(graph, th.tensor([edge_id]))
        test_edges.append(TestEdge(src, dst, 0))

    graph.remove_edges(sampled_edge_ids)
    test_graph = test_edges

    return graph, test_graph
