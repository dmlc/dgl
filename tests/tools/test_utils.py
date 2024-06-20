import logging
import os
import tempfile

import numpy as np
import pyarrow
import pytest

from distpartitioning.utils import (
    constants,
    get_edge_types,
    get_etype_featnames,
    get_gid_offsets,
    get_gnid_range_map,
    get_node_types,
    get_ntype_featnames,
    read_json,
)


def test_read_json():
    """Unit test for the function read_json in the
    tools/distpartitioning/utils.py module
    """

    # generate input test file.
    json_str = '{"name" : "test_file_name"}'

    with tempfile.TemporaryDirectory() as root_dir:
        # create input json file
        input_test_file = os.path.join(root_dir, "metadata.json")

        input_fd = open(input_test_file, "w")
        input_fd.write(json_str)
        input_fd.close()

        # begin test
        actual_output = read_json(input_test_file)

        assert len(actual_output) == 1
        assert actual_output["name"] == "test_file_name"


def test_get_etype_featnames():
    """Unit test for the funciton get_etype_featnames in the
    tools/distpartitioning/utils.py module
    """
    format_dict = {"name": "numpy"}

    edge_feat_data = {}
    for idx in range(1, 10):
        edge_feat_data[f"edge_feat_{idx}"] = {}
        edge_feat_data[f"edge_feat_{idx}"]["format"] = format_dict
        edge_feat_data[f"edge_feat_{idx}"]["data"] = []

    edge_data = {}
    edge_data["n1:e1:n1"] = edge_feat_data

    schema_dict = {}
    schema_dict[constants.STR_EDGE_DATA] = edge_data

    # test the function
    actual_feat_names = get_etype_featnames("n1:e1:n1", schema_dict)

    # assert here
    for idx in range(1, 10):
        assert f"edge_feat_{idx}" in actual_feat_names


def test_get_ntype_featnames():
    """Unit test for the function get_ntype_featnames in the
    tools/distpartitioning/utils.py module
    """
    # prepare test data
    node_feat_data = {}
    for idx in range(1, 10):
        node_feat_data[f"node_feat_{idx}"] = {}

    node_data = {}
    node_data["n1"] = node_feat_data

    schema_dict = {}
    schema_dict[constants.STR_NODE_DATA] = node_data

    # test
    actual_feat_names = get_ntype_featnames("n1", schema_dict)

    # assert here
    for idx in range(1, 10):
        assert f"node_feat_{idx}" in actual_feat_names


def test_get_edge_types():
    """Unit test for the function get_edge_types in the
    tools/distpartitioning/utils.py module
    """

    # Prepare test data
    edge_types = [f"edge_type_{idx}" for idx in range(10)]
    schema_dict = {}
    schema_dict[constants.STR_EDGE_TYPE] = edge_types

    # Test
    etype_etypeid_map, etypes, etypeid_etype_map = get_edge_types(schema_dict)

    # assert
    for idx in range(10):
        assert f"edge_type_{idx}" in edge_types
        assert etype_etypeid_map[f"edge_type_{idx}"] == idx
        assert etypeid_etype_map[idx] == f"edge_type_{idx}"


def test_get_node_types():
    """Unit test for the function get_node_types in the
    tools/distpartitioning/utils.py module
    """

    # Prepare test data
    node_types = [f"node_type_{idx}" for idx in range(10)]
    schema_dict = {}
    schema_dict[constants.STR_NODE_TYPE] = node_types

    # Test
    ntype_ntypeid_map, ntypes, ntypeid_ntype_map = get_node_types(schema_dict)

    # assert results here
    for idx in range(10):
        assert f"node_type_{idx}" in node_types
        assert ntype_ntypeid_map[f"node_type_{idx}"] == idx
        assert ntypeid_ntype_map[idx] == f"node_type_{idx}"


def test_get_gid_offsets():
    """Unit test for the function get_gid_offsets in the
    tools/distpartitioning/utils.py module
    typenames, typecounts
    """

    # Prepare test data
    node_types = [f"node_type_{idx}" for idx in range(10)]
    ntype_counts = {}
    for idx in range(10):
        ntype_counts[f"node_type_{idx}"] = idx * 100

    # Test
    actual_offsets = get_gid_offsets(node_types, ntype_counts)

    # assert on the results
    offset = 0
    for idx in range(10):
        [start, end] = actual_offsets[f"node_type_{idx}"]
        assert start == offset
        assert end == (offset + idx * 100)
        offset += idx * 100


def test_get_gnid_range_map():
    """Unit test for the function get_gnid_range_map in the
    tools/distpartitioning/utils.py module
    """
    # Prepare test data
    node_tids = {}
    for idx in range(5):
        tids = []
        for tt in range(idx + 1):
            tids.append([tt * 10, (tt + 1) * 10])
        node_tids[f"node_type_{idx}"] = tids

    # Test
    ntypes_gid_range = get_gnid_range_map(node_tids)

    # assert the results
    offset = 0
    for idx in range(5):
        assert f"node_type_{idx}" in ntypes_gid_range
        val = ntypes_gid_range[f"node_type_{idx}"]
        assert val[0] == offset
        assert val[1] == offset + (idx + 1) * 10
        offset = val[1]
