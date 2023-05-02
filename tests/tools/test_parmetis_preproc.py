import os
import tempfile
from collections import namedtuple

import numpy as np
import pytest
from distpartitioning import array_readwriter, constants
from distpartitioning.parmetis_preprocess import gen_edge_files
from distpartitioning.utils import generate_roundrobin_read_list
from numpy.testing import assert_array_equal

NODE_TYPE = "n1"
EDGE_TYPE = f"{NODE_TYPE}:e1:{NODE_TYPE}"


def _read_file(fname, fmt_name, fmt_delimiter):
    """Read a file

    Parameters:
    -----------
    fname : string
        filename of the input file to read
    fmt_name : string
        specifying whether it is a csv or a parquet file
    fmt_delimiter : string
        string specifying the delimiter used in the input file
    """
    reader_fmt_meta = {
        "name": fmt_name,
    }
    if fmt_name == constants.STR_CSV:
        reader_fmt_meta["delimiter"] = fmt_delimiter
    data_df = array_readwriter.get_array_parser(**reader_fmt_meta).read(fname)
    return data_df


def _get_test_data(edges_dir, num_chunks, edge_fmt, edge_fmt_del):
    """Creates unit test input which are a set of edge files
    in the following format "src_node_id<delimiter>dst_node_id"

    Parameters:
    -----------
    edges_dir : str
        folder where edge files are stored
    num_chunks : int
        no. of files to create for each edge type
    edge_fmt : str, optional
        to specify whether this file is csv or parquet
    edge_fmt_del : str optional
        delimiter to use in the edges file

    Returns:
    --------
    dict :
        dictionary created which represents the schema used for
        creating the input dataset
    """
    schema = {}
    schema["num_nodes_per_type"] = [10]
    schema["edge_type"] = [EDGE_TYPE]
    schema["node_type"] = [NODE_TYPE]

    edges = {}
    edges[EDGE_TYPE] = {}
    edges[EDGE_TYPE]["format"] = {}
    edges[EDGE_TYPE]["format"]["name"] = edge_fmt
    edges[EDGE_TYPE]["format"]["delimiter"] = edge_fmt_del

    os.makedirs(edges_dir, exist_ok=True)
    fmt_meta = {"name": edge_fmt}
    if edge_fmt == "csv":
        fmt_meta["delimiter"] = edge_fmt_del

    edge_files = []
    for idx in range(num_chunks):
        path = os.path.join(edges_dir, f"test_file_{idx}.{fmt_meta['name']}")
        array_parser = array_readwriter.get_array_parser(**fmt_meta)
        edge_data = (
            np.array([np.arange(10), np.arange(10)]).reshape(10, 2) + 10 * idx
        )
        array_parser.write(path, edge_data)

        edge_files.append(path)

    edges[EDGE_TYPE]["data"] = edge_files
    schema["edges"] = edges

    return schema


@pytest.mark.parametrize("num_chunks, num_parts", [[4, 1], [4, 2], [4, 4]])
@pytest.mark.parametrize("edges_fmt", ["csv", "parquet"])
@pytest.mark.parametrize("edges_delimiter", [" ", ","])
def test_gen_edge_files(num_chunks, num_parts, edges_fmt, edges_delimiter):
    """Unit test case for the function
    tools/distpartitioning/parmetis_preprocess.py::gen_edge_files

    Parameters:
    -----------
    num_chunks : int
        no. of chunks the input graph needs to be partititioned into
    num_parts : int
        no. of partitions
    edges_fmt : string
        specifying the storage format for the edge files
    edges_delimiter : string
        specifying the delimiter used in the edge files
    """
    # Create the input dataset
    with tempfile.TemporaryDirectory() as root_dir:

        # Create expected environment for test
        input_dir = os.path.join(root_dir, "chunked-data")
        output_dir = os.path.join(root_dir, "preproc_dir")

        # Mock a parser object
        fn_params = namedtuple("fn_params", "input_dir output_dir num_parts")
        fn_params.input_dir = input_dir
        fn_params.output_dir = output_dir
        fn_params.num_parts = num_parts

        # Create test files and get corresponding file schema
        schema_map = _get_test_data(
            input_dir, num_chunks, edges_fmt, edges_delimiter
        )
        edges_file_list = schema_map["edges"][EDGE_TYPE]["data"]
        # This is breaking encapsulation, but no other good way to get file list
        rank_assignments = generate_roundrobin_read_list(
            len(edges_file_list), num_parts
        )

        # Get the global node id offsets for each node type
        # There is only one node-type in the test graph
        # which range from 0 thru 9.
        ntype_gnid_offset = {}
        ntype_gnid_offset[NODE_TYPE] = np.array([0, 10 * num_chunks]).reshape(
            1, 2
        )

        # Iterate over no. of partitions
        for rank in range(num_parts):
            actual_results = gen_edge_files(rank, schema_map, fn_params)

            # Get the original files
            original_files = [
                edges_file_list[file_idx] for file_idx in rank_assignments[rank]
            ]

            # Validate the results with the baseline results
            # Test 1. no. of files should have the same count per rank
            assert len(original_files) == len(actual_results)
            assert len(actual_results) > 0

            # Test 2. Check the contents of each file and verify the
            # file contents match with the expected results.
            for actual_fname, original_fname in zip(
                actual_results, original_files
            ):
                # Check the actual file exists
                assert os.path.isfile(actual_fname)
                # Read both files and compare the edges
                # Here note that the src and dst end points are global_node_ids
                actual_data = _read_file(actual_fname, "csv", " ")
                expected_data = _read_file(
                    original_fname, edges_fmt, edges_delimiter
                )

                # Subtract the global node id offsets, so that we get type node ids
                # In the current unit test case, the graph has only one node-type.
                # and this means that type-node-ids are same as the global-node-ids.
                # Below two lines will take take into effect when the graphs have
                # more than one node type.
                actual_data[:, 0] -= ntype_gnid_offset[NODE_TYPE][0, 0]
                actual_data[:, 1] -= ntype_gnid_offset[NODE_TYPE][0, 0]

                # Verify that the contents are equal
                assert_array_equal(expected_data, actual_data)
