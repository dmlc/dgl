# Requires setting PYTHONPATH=${GITROOT}/tools
import argparse
import json
import logging
import os

import numpy as np
from base import dump_partition_meta, PartitionMeta
from distpartitioning import array_readwriter
from files import setdir


def _random_partition(metadata, num_parts):
    num_nodes_per_type = metadata["num_nodes_per_type"]
    ntypes = metadata["node_type"]
    for ntype, n in zip(ntypes, num_nodes_per_type):
        logging.info("Generating partition for node type %s" % ntype)
        parts = np.random.randint(0, num_parts, (n,))
        array_readwriter.get_array_parser(name="csv").write(
            ntype + ".txt", parts
        )


def random_partition(metadata, num_parts, output_path):
    """
    Randomly partition the graph described in metadata and generate partition ID mapping
    in :attr:`output_path`.

    A directory will be created at :attr:`output_path` containing the partition ID
    mapping files named "<node-type>.txt" (e.g. "author.txt", "paper.txt" and
    "institution.txt" for OGB-MAG240M).  Each file contains one line per node representing
    the partition ID the node belongs to.
    In addition, metadata which includes version, number of partitions is dumped.
    """
    with setdir(output_path):
        _random_partition(metadata, num_parts)
        part_meta = PartitionMeta(
            version="1.0.0", num_parts=num_parts, algo_name="random"
        )
        dump_partition_meta(part_meta, "partition_meta.json")


# Run with PYTHONPATH=${GIT_ROOT_DIR}/tools
# where ${GIT_ROOT_DIR} is the directory to the DGL git repository.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_dir",
        type=str,
        help="input directory that contains the metadata file",
    )
    parser.add_argument("--out_dir", type=str, help="output directory")
    parser.add_argument(
        "--num_partitions", type=int, help="number of partitions"
    )
    logging.basicConfig(level="INFO")
    args = parser.parse_args()
    with open(os.path.join(args.in_dir, "metadata.json")) as f:
        metadata = json.load(f)
    num_parts = args.num_partitions
    random_partition(metadata, num_parts, args.out_dir)
