import json
import logging
import sys
import os
import numpy as np

from utils import setdir
import array_readwriter

def _random_partition(metadata, num_parts):
    num_nodes_per_type = [sum(_) for _ in metadata['num_nodes_per_chunk']]
    ntypes = metadata['node_type']
    for ntype, n in zip(ntypes, num_nodes_per_type):
        logging.info('Generating partition for node type %s' % ntype)
        parts = np.random.randint(0, num_parts, (n,))
        array_readwriter.write(ntype + '.txt', parts, name='csv')

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

if __name__ == '__main__':
    logging.basicConfig(level='INFO')
    with open(sys.argv[1]) as f:
        metadata = json.load(f)
    output_path = sys.argv[2]
    num_parts = int(sys.argv[3])
    random_partition(metadata, num_parts, output_path)
