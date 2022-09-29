import argparse
import json
import logging
import os
import platform
import sys
import tempfile

import dgl
import numpy as np
import torch
from chunk_graph import chunk_graph
from dgl.data.utils import load_graphs, load_tensors

from create_chunked_dataset import create_chunked_dataset


def test_parmetis_preprocessing():

    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        num_nodes = 720
        num_institutions = 1200
        num_authors = 1200
        num_papers = 1200

        all_ntypes, all_etypes, _ = create_chunked_dataset(
            root_dir, num_chunks, include_masks=True
        )

        # Generate random parmetis partition ids for the nodes in the graph.
        # Replace this code with actual ParMETIS executable when it is ready
        output_dir = os.path.join(root_dir, 'chunked-data')
        parmetis_file = os.path.join(output_dir, 'parmetis_output.txt')
        node_ids = np.arange(num_nodes)
        partition_ids = np.random.randint(0, 2, (num_nodes,))
        parmetis_output = np.column_stack([node_ids, partition_ids])

        # Create parmetis output, this is mimicking running actual parmetis.
        with open(parmetis_file, 'w') as f:
            np.savetxt(f, parmetis_output)

        # Check the post processing script here.
        results_dir = os.path.join(output_dir, 'partitions_dir')
        json_file = os.path.join(output_dir, 'metadata.json')
        print(json_file)
        print(results_dir)
        print(parmetis_file)
        os.system(
            f'python3 tools/distpartitioning/parmetis_postprocess.py '
            f'--schema_file {json_file} '
            f'--parmetis_output_file {parmetis_file} '
            f'--partitions_dir {results_dir}'
        )

        ntype_count = {
            'author': num_authors,
            'paper': num_papers,
            'institution': num_institutions,
        }
        for ntype_name in ['author', 'paper', 'institution']:
            fname = os.path.join(results_dir, f'{ntype_name}.txt')
            print(fname)
            assert os.path.isfile(fname)

            # Load and check the partition ids in this file.
            part_ids = np.loadtxt(fname)
            assert part_ids.shape[0] == ntype_count[ntype_name]
            assert np.min(part_ids) == 0
            assert np.max(part_ids) == 1


if __name__ == '__main__':
    # Configure logging.
    logging.basicConfig(
        level='INFO',
        format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )
    test_parmetis_preprocessing()
