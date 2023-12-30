import json
import os
import tempfile
import unittest
from collections import Counter

import dgl

import pytest
from change_etype_to_canonical_etype import convert_conf, is_old_version
from dgl.distributed import partition_graph
from scipy import sparse as spsp


def create_random_hetero(type_n, node_n):
    num_nodes = {}
    for i in range(1, type_n + 1):
        num_nodes[f"n{i}"] = node_n
    c_etypes = []
    count = 0
    for i in range(1, type_n):
        for j in range(i + 1, type_n + 1):
            count += 1
            c_etypes.append((f"n{i}", f"r{count}", f"n{j}"))
    edges = {}
    for etype in c_etypes:
        src_ntype, _, dst_ntype = etype
        arr = spsp.random(
            num_nodes[src_ntype],
            num_nodes[dst_ntype],
            density=0.001,
            format="coo",
            random_state=100,
        )
        edges[etype] = (arr.row, arr.col)
    return dgl.heterograph(edges, num_nodes), [
        ":".join(c_etype) for c_etype in c_etypes
    ]


@unittest.skip(reason="Skip due to glitch in CI")
@pytest.mark.parametrize(
    "type_n, node_n, num_parts", [[3, 100, 2], [10, 500, 4], [10, 1000, 8]]
)
def test_hetero_graph(type_n, node_n, num_parts):
    g, expected_c_etypes = create_random_hetero(type_n, node_n)
    do_convert_and_check(g, "convert_conf_test", num_parts, expected_c_etypes)


@unittest.skip(reason="Skip due to glitch in CI")
@pytest.mark.parametrize("node_n, num_parts", [[100, 2], [500, 4]])
def test_homo_graph(node_n, num_parts):
    g = dgl.rand_graph(node_n, node_n // 10)
    do_convert_and_check(g, "convert_conf_test", num_parts, ["_N:_E:_N"])


def do_convert_and_check(g, graph_name, num_parts, expected_c_etypes):
    with tempfile.TemporaryDirectory() as root_dir:
        partition_graph(g, graph_name, num_parts, root_dir)
        part_config = os.path.join(root_dir, graph_name + ".json")
        old_config = _get_old_config(part_config)
        # Call convert function
        convert_conf(part_config)
        with open(part_config, "r") as config_f:
            config = json.load(config_f)
            # Check we get all canonical etypes
            assert Counter(expected_c_etypes) == Counter(
                config["etypes"].keys()
            )
            # Check the id is match after transform from etypes -> canonical
            assert old_config["etypes"] == _extract_etypes(config["etypes"])


def _get_old_config(part_config):
    with open(part_config, "r+") as config_f:
        config = json.load(config_f)
        if not is_old_version(config):
            config["etypes"] = _extract_etypes(config["etypes"])
            config["edge_map"] = _extract_edge_map(config["edge_map"])
            config_f.seek(0)
            json.dump(config, config_f, indent=4)
            config_f.truncate()
        return config


def _extract_etypes(c_etypes):
    etypes = {}
    for c_etype, eid in c_etypes.items():
        etype = c_etype.split(":")[1]
        etypes[etype] = eid
    return etypes


def _extract_edge_map(c_edge_map):
    edge_map = {}
    for c_etype, emap in c_edge_map.items():
        etype = c_etype.split(":")[1]
        edge_map[etype] = emap
    return edge_map
