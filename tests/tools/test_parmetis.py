import argparse
import json
import os
import sys
import tempfile
import unittest

import dgl
import numpy as np
import torch
from dgl.data.utils import load_graphs, load_tensors
from partition_algo.base import load_partition_meta

from pytest_utils import create_chunked_dataset

"""
TODO: skipping this test case since the dependency, mpirun, is
not yet configured in the CI framework.
"""


@unittest.skipIf(True, reason="mpi is not available in CI test framework.")
def test_parmetis_preprocessing():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        g = create_chunked_dataset(root_dir, num_chunks)

        # Trigger ParMETIS pre-processing here.
        input_dir = os.path.join(root_dir, "chunked-data")
        results_dir = os.path.join(root_dir, "parmetis-data")
        os.system(
            f"mpirun -np {num_chunks} python3 tools/distpartitioning/parmetis_preprocess.py "
            f"--schema {metadata.json} "
            f"--input_dir {input_dir} "
            f"--output_dir {results_dir} "
            f"--num_parts {num_chunks}"
        )

        # Now add all the tests and check whether the test has passed or failed.
        # Read parmetis_nfiles and ensure all files are present.
        parmetis_data_dir = os.path.join(root_dir, "parmetis-data")
        assert os.path.isdir(parmetis_data_dir)
        parmetis_nodes_file = os.path.join(
            parmetis_data_dir, "parmetis_nfiles.txt"
        )
        assert os.path.isfile(parmetis_nodes_file)

        # `parmetis_nfiles.txt` should have each line in the following format.
        # <filename> <global_id_start> <global_id_end>
        with open(parmetis_nodes_file, "r") as nodes_metafile:
            lines = nodes_metafile.readlines()
            total_node_count = 0
            for line in lines:
                tokens = line.split(" ")
                assert len(tokens) == 3
                assert os.path.isfile(tokens[0])
                assert int(tokens[1]) == total_node_count

                # check contents of each of the nodes files here
                with open(tokens[0], "r") as nodes_file:
                    node_lines = nodes_file.readlines()
                    for line in node_lines:
                        val = line.split(" ")
                        # <ntype_id> <weight_list> <mask_list> <type_node_id>
                        assert len(val) == 8
                    node_count = len(node_lines)
                    total_node_count += node_count
                assert int(tokens[2]) == total_node_count

        # Meta_data object.
        output_dir = os.path.join(root_dir, "chunked-data")
        json_file = os.path.join(output_dir, "metadata.json")
        assert os.path.isfile(json_file)
        with open(json_file, "rb") as f:
            meta_data = json.load(f)

        # Count the total no. of nodes.
        true_node_count = 0
        num_nodes_per_chunk = meta_data["num_nodes_per_chunk"]
        for i in range(len(num_nodes_per_chunk)):
            node_per_part = num_nodes_per_chunk[i]
            for j in range(len(node_per_part)):
                true_node_count += node_per_part[j]
        assert total_node_count == true_node_count

        # Read parmetis_efiles and ensure all files are present.
        # This file contains a list of filenames.
        parmetis_edges_file = os.path.join(
            parmetis_data_dir, "parmetis_efiles.txt"
        )
        assert os.path.isfile(parmetis_edges_file)

        with open(parmetis_edges_file, "r") as edges_metafile:
            lines = edges_metafile.readlines()
            total_edge_count = 0
            for line in lines:
                edges_filename = line.strip()
                assert os.path.isfile(edges_filename)

                with open(edges_filename, "r") as edges_file:
                    edge_lines = edges_file.readlines()
                    total_edge_count += len(edge_lines)
                    for line in edge_lines:
                        val = line.split(" ")
                        assert len(val) == 2

        # Count the total no. of edges
        true_edge_count = 0
        num_edges_per_chunk = meta_data["num_edges_per_chunk"]
        for i in range(len(num_edges_per_chunk)):
            edges_per_part = num_edges_per_chunk[i]
            for j in range(len(edges_per_part)):
                true_edge_count += edges_per_part[j]
        assert true_edge_count == total_edge_count


def test_parmetis_postprocessing():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        g = create_chunked_dataset(root_dir, num_chunks)

        num_nodes = g.num_nodes()
        num_institutions = g.num_nodes("institution")
        num_authors = g.num_nodes("author")
        num_papers = g.num_nodes("paper")

        # Generate random parmetis partition ids for the nodes in the graph.
        # Replace this code with actual ParMETIS executable when it is ready
        output_dir = os.path.join(root_dir, "chunked-data")
        assert os.path.isdir(output_dir)

        parmetis_file = os.path.join(output_dir, "parmetis_output.txt")
        node_ids = np.arange(num_nodes)
        partition_ids = np.random.randint(0, 2, (num_nodes,))
        parmetis_output = np.column_stack([node_ids, partition_ids])

        # Create parmetis output, this is mimicking running actual parmetis.
        with open(parmetis_file, "w") as f:
            np.savetxt(f, parmetis_output)
        assert os.path.isfile(parmetis_file)

        # Check the post processing script here.
        results_dir = os.path.join(output_dir, "partitions_dir")
        json_file = os.path.join(output_dir, "metadata.json")
        print(json_file)
        print(results_dir)
        print(parmetis_file)
        os.system(
            f"python3 tools/distpartitioning/parmetis_postprocess.py "
            f"--postproc_input_dir {output_dir} "
            f"--schema_file metadata.json "
            f"--parmetis_output_file {parmetis_file} "
            f"--partitions_dir {results_dir}"
        )

        ntype_count = {
            "author": num_authors,
            "paper": num_papers,
            "institution": num_institutions,
        }
        for ntype_name in ["author", "paper", "institution"]:
            fname = os.path.join(results_dir, f"{ntype_name}.txt")
            print(fname)
            assert os.path.isfile(fname)

            # Load and check the partition ids in this file.
            part_ids = np.loadtxt(fname)
            assert part_ids.shape[0] == ntype_count[ntype_name]
            assert np.min(part_ids) == 0
            assert np.max(part_ids) == 1

        # check partition meta file
        part_meta_file = os.path.join(results_dir, "partition_meta.json")
        assert os.path.isfile(part_meta_file)
        part_meta = load_partition_meta(part_meta_file)
        assert part_meta.num_parts == 2
        assert part_meta.algo_name == "metis"


"""
TODO: skipping this test case since it depends on the dependency, mpi,
which is not yet configured in the CI framework.
"""


@unittest.skipIf(True, reason="mpi is not available in CI test framework.")
def test_parmetis_wrapper():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        graph_name = "mag240m"
        g = create_chunked_dataset(root_dir, num_chunks)
        all_ntypes = g.ntypes
        all_etypes = g.etypes
        num_constraints = len(all_ntypes) + 3
        num_institutions = g.num_nodes("institution")
        num_authors = g.num_nodes("author")
        num_papers = g.num_nodes("paper")

        # Trigger ParMETIS.
        schema_file = os.path.join(root_dir, "chunked-data/metadata.json")
        preproc_input_dir = os.path.join(root_dir, "chunked-data")
        preproc_output_dir = os.path.join(
            root_dir, "chunked-data/preproc_output_dir"
        )
        parmetis_output_file = os.path.join(
            os.getcwd(), f"{graph_name}_part.{num_chunks}"
        )
        partitions_dir = os.path.join(root_dir, "chunked-data/partitions_dir")
        hostfile = os.path.join(root_dir, "ip_config.txt")
        with open(hostfile, "w") as f:
            f.write("127.0.0.1\n")
            f.write("127.0.0.1\n")

        num_nodes = g.num_nodes()
        num_edges = g.num_edges()
        stats_file = f"{graph_name}_stats.txt"
        with open(stats_file, "w") as f:
            f.write(f"{num_nodes} {num_edges} {num_constraints}")

        os.system(
            f"python3 tools/distpartitioning/parmetis_wrapper.py "
            f"--schema_file {schema_file} "
            f"--preproc_input_dir {preproc_input_dir} "
            f"--preproc_output_dir {preproc_output_dir} "
            f"--hostfile {hostfile} "
            f"--num_parts {num_chunks} "
            f"--parmetis_output_file {parmetis_output_file} "
            f"--partitions_dir {partitions_dir} "
        )
        print("Executing Done.")

        ntype_count = {
            "author": num_authors,
            "paper": num_papers,
            "institution": num_institutions,
        }
        for ntype_name in ["author", "paper", "institution"]:
            fname = os.path.join(partitions_dir, f"{ntype_name}.txt")
            print(fname)
            assert os.path.isfile(fname)

            # Load and check the partition ids in this file.
            part_ids = np.loadtxt(fname)
            assert part_ids.shape[0] == ntype_count[ntype_name]
            assert np.min(part_ids) == 0
            assert np.max(part_ids) == (num_chunks - 1)
