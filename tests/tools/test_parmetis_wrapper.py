import argparse
import dgl
import json
import numpy as np
import os
import sys
from pathlib import Path
import tempfile
import torch
import logging
import platform

from dgl.data.utils import load_tensors, load_graphs
from chunk_graph import chunk_graph
from create_chunked_dataset import create_chunked_dataset


def test_parmetis_wrapper():
    with tempfile.TemporaryDirectory() as root_dir:
        num_chunks = 2
        graph_name = "mag240m"
        g = create_chunked_dataset(root_dir, num_chunks, include_masks=True)
        all_ntypes = g.ntypes
        all_etypes = g.etypes
        num_constraints = len(all_ntypes) + 3
        num_institutions = g.number_of_nodes('institution')
        num_authors = g.number_of_nodes('author')
        num_papers = g.number_of_nodes('paper')

        # Trigger ParMETIS.
        schema_file = os.path.join(root_dir, 'chunked-data/metadata.json')
        preproc_output_dir = os.path.join(
            root_dir, 'chunked-data/preproc_output_dir'
        )
        parmetis_output_file = os.path.join(
            os.getcwd(), f'{graph_name}_part.{num_chunks}'
        )
        partitions_dir = os.path.join(root_dir, 'chunked-data/partitions_dir')
        hostfile = os.path.join(root_dir, 'ip_config.txt')
        with open(hostfile, 'w') as f:
            f.write('127.0.0.1\n')
            f.write('127.0.0.1\n')

        num_nodes = g.number_of_nodes()
        num_edges = g.number_of_edges()
        stats_file = f'{graph_name}_stats.txt'
        with open(stats_file, 'w') as f:
            f.write(f'{num_nodes} {num_edges} {num_constraints}')

        parmetis_cmd = (
            f'python3 tools/distpartitioning/parmetis_wrapper.py '
            f'--schema_file {schema_file} '
            f'--preproc_output_dir {preproc_output_dir} '
            f'--hostfile {hostfile} '
            f'--parmetis_output_file {parmetis_output_file} '
            f'--partitions_dir {partitions_dir} '
        )
        logging.info(f'Executing the following cmd: {parmetis_cmd}')
        print(parmetis_cmd)
        os.system(parmetis_cmd)

        ntype_count = {
            'author': num_authors,
            'paper': num_papers,
            'institution': num_institutions,
        }
        for ntype_name in ['author', 'paper', 'institution']:
            fname = os.path.join(partitions_dir, f'{ntype_name}.txt')
            print(fname)
            assert os.path.isfile(fname)

            # Load and check the partition ids in this file.
            part_ids = np.loadtxt(fname)
            assert part_ids.shape[0] == ntype_count[ntype_name]
            assert np.min(part_ids) == 0
            assert np.max(part_ids) == (num_chunks - 1)


if __name__ == '__main__':
    # Configure logging.
    logging.basicConfig(
        level='INFO',
        format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )
    test_parmetis_wrapper()
