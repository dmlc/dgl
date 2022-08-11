# Requires setting PYTHONPATH=${GITROOT}/tools
import json
import logging
import sys
import os
import numpy as np
import argparse

from utils import setdir
from utils import array_readwriter

def _random_partition(metadata, num_parts):
    num_nodes_per_type = [sum(_) for _ in metadata['num_nodes_per_chunk']]
    ntypes = metadata['node_type']
    for ntype, n in zip(ntypes, num_nodes_per_type):
        logging.info('Generating partition for node type %s' % ntype)
        parts = np.random.randint(0, num_parts, (n,))
        array_readwriter.get_array_parser(name='csv').write(ntype + '.txt', parts)

def random_partition(metadata, num_parts, output_path):
    """
    Randomly partition the graph described in metadata and generate partition ID mapping
    in :attr:`output_path`.

    A directory will be created at :attr:`output_path` containing the partition ID
    mapping files named "<node-type>.txt" (e.g. "author.txt", "paper.txt" and
    "institution.txt" for OGB-MAG240M).  Each file contains one line per node representing
    the partition ID the node belongs to.
    """
    with setdir(output_path):
        _random_partition(metadata, num_parts)

# Run with PYTHONPATH=${GIT_ROOT_DIR}/tools
# where ${GIT_ROOT_DIR} is the directory to the DGL git repository.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            'metadata', type=str, help='input metadata file of the chunked graph format')
    parser.add_argument(
            'output_path', type=str, help='output directory')
    parser.add_argument(
            'num_partitions', type=int, help='number of partitions')
    logging.basicConfig(level='INFO')
    args = parser.parse_args()
    with open(args.metadata) as f:
        metadata = json.load(f)
    output_path = args.output_path
    num_parts = args.num_partitions
    random_partition(metadata, num_parts, output_path)
