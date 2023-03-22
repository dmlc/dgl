import argparse
import json
import logging
import os
import platform
import sys
import tempfile

import numpy as np
import pytest
import torch
from distpartitioning import array_readwriter, constants
from distpartitioning.parmetis_preprocess import gen_edge_files
from distpartitioning.utils import generate_read_list, get_idranges, read_json
from partition_algo.base import load_partition_meta
from pytest_utils import create_chunked_dataset


def _get_parse_args_gen_edge_files(args):
    """Create argparser object

    Parameters:
    -----------
    args : string
        string specifying the command line arguments for parmetis_preprocess.py

    Returns:
    --------
    argparser object :
        instance of argparser object to capture the command line arguments
    """
    parser = argparse.ArgumentParser(description="Testing parmetis preproc")
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
    )
    parser.add_argument(
        "--num_parts",
        required=True,
        type=int,
    )
    return parser.parse_args(args)


def _get_edge_files(schema_map, rank, num_parts):
    """Returns the edge files processed by each rank

    Parameters:
    ----------
    schema_map : dict, json
        dictionary object created by reading the input graph's
        metadata.json file
    rank : integer
        specifying the rank of the process
    num_parts : integer
        no. of partitions for the input graph

    Returns:
    --------
    list, string :
        specifying the edge file names
    list, tuples :
        each tuple containing edge file type and delimiter used in the
        corresponding edge file
    list, string :
        specifying the edge type for each of the edge files
    """
    edge_data = schema_map[constants.STR_EDGES]
    etype_names = schema_map[constants.STR_EDGE_TYPE]

    edge_files = []  # used for file names
    meta_files = []  # used for storing file types and delimiter
    edge_types = []  # used for storing the edge type name

    # Iterate over the `edges` key in the input metadata
    # its value is a dictionary whose keys are edge names
    # and value is a dictionary as well.
    for etype_name, etype_info in edge_data.items():

        # Get the list of files for this edge type
        edge_data_files = etype_info[constants.STR_DATA]
        # Get the file type, 'csv' or 'parquet'
        edges_format = etype_info[constants.STR_FORMAT][constants.STR_NAME]

        # Delimiter used for the edge files
        edges_delimiter = None
        if edges_format == constants.STR_CSV:
            edges_delimiter = etype_info[constants.STR_FORMAT][
                constants.STR_FORMAT_DELIMITER
            ]

        # Split the files among the no. of workers
        file_idxes = generate_read_list(len(edge_data_files), num_parts)
        for idx in file_idxes[rank]:
            edge_files.append(edge_data_files[idx])
            meta_files.append((edges_format, edges_delimiter))
            edge_types.append(etype_name)

    # Return the edge file names, format information and file types
    return edge_files, meta_files, edge_types


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


@pytest.mark.parametrize("num_chunks, num_parts", [[4, 1], [4, 2], [4, 4]])
@pytest.mark.parametrize("edges_fmt", ["csv", "parquet"])
def test_gen_edge_files(num_chunks, num_parts, edges_fmt):
    """Unit test case for the function `tools/distpartitioning/parmetis_preprocess.py::gen_edge_files

    Parameters:
    -----------
    num_chunks : int
        no. of chunks the input graph needs to be partititioned into
    num_parts : int
        no. of partitions
    edges_fmt : string
        specifying the storage format for the edge files
    """

    # Create the input dataset
    with tempfile.TemporaryDirectory() as root_dir:
        g = create_chunked_dataset(
            root_dir, num_chunks, data_fmt="numpy", edges_fmt=edges_fmt
        )

        # Prepare the state information for firing unit test
        input_dir = os.path.join(root_dir, "chunked-data")
        metadata_path = os.path.join(input_dir, "metadata.json")
        assert os.path.isfile(
            metadata_path
        ), f"Metadata json file does not exist"
        output_dir = os.path.join(input_dir, "preproc_dir")

        # Get the parser object
        args = f"--input_dir {input_dir} --output_dir {output_dir} --num_parts {num_parts}".split()
        parsed_args = _get_parse_args_gen_edge_files(args)

        # Read the input schema
        schema_map = read_json(metadata_path)

        # Get the global node id offsets for each node type
        _, ntype_gnid_offset = get_idranges(
            schema_map[constants.STR_NODE_TYPE],
            dict(
                zip(
                    schema_map[constants.STR_NODE_TYPE],
                    schema_map[constants.STR_NUM_NODES_PER_TYPE],
                )
            ),
        )

        # Iterate over no. of partitions
        for rank in range(num_parts):

            # Fire the unit test case and get the results
            actual_results = gen_edge_files(rank, schema_map, parsed_args)

            # Validate the outputs of this method.
            logging.info(f"Validating the results for Rank: {rank}")

            # Get the gold results for baseline comparision
            baseline_results, fmt_results, edge_types = _get_edge_files(
                schema_map, rank, num_parts
            )

            # Validate the results with the baseline results
            # Test 1. no. of files should have the same count per rank
            assert len(baseline_results) == len(actual_results)

            # Test 2. Check the contents of each file and verify the
            # file contents
            for idx, fname in enumerate(baseline_results):

                # Check the file exists
                assert os.path.isfile(fname)
                edge_file = fname.split("/")[-1]

                # ``edgetype`` strings are in canonical format, src_node_type:edge_type:dst_node_type
                tokens = edge_types[idx].split(":")
                assert len(tokens) == 3

                src_ntype_name = tokens[0]
                rel_name = tokens[1]
                dst_ntype_name = tokens[2]

                # Read both files and compare the edges
                # Here note that the src and dst end points are global_node_ids
                target_file = os.path.join(output_dir, f"edges_{edge_file}")
                target_data = _read_file(target_file, constants.STR_CSV, " ")

                # Subtract the global node id offsets, so that we get type node ids
                target_data[:, 0] -= ntype_gnid_offset[src_ntype_name][0, 0]
                target_data[:, 1] -= ntype_gnid_offset[dst_ntype_name][0, 0]

                # Now compare with the edge files from the input dataset
                fmt_type = fmt_results[idx][0]
                fmt_delimiter = fmt_results[idx][1]
                source_file = os.path.join(
                    os.path.join(input_dir, "edge_index"), edge_file
                )
                logging.info(
                    f"SourceFile: {source_file}, TargetFile: {target_file}"
                )
                source_data = _read_file(source_file, fmt_type, fmt_delimiter)

                # Verify that the contents are equal
                assert np.all(target_data == source_data)
