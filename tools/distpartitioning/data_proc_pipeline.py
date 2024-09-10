import argparse
import logging
import os
import platform

import numpy as np
import torch.multiprocessing as mp

from data_shuffle import multi_machine_run, single_machine_run


def log_params(params):
    """Print all the command line arguments for debugging purposes.

    Parameters:
    -----------
    params: argparse object
        Argument Parser structure listing all the pre-defined parameters
    """
    print("Input Dir: ", params.input_dir)
    print("Graph Name: ", params.graph_name)
    print("Schema File: ", params.schema)
    print("No. partitions: ", params.num_parts)
    print("Output Dir: ", params.output)
    print("WorldSize: ", params.world_size)
    print("Metis partitions: ", params.partitions_dir)


if __name__ == "__main__":
    """
    Start of execution from this point.
    Invoke the appropriate function to begin execution
    """
    # arguments which are already needed by the existing implementation of convert_partition.py
    parser = argparse.ArgumentParser(description="Construct graph partitions")
    parser.add_argument(
        "--input-dir",
        required=True,
        type=str,
        help="The directory path that contains the partition results.",
    )
    parser.add_argument(
        "--graph-name", required=True, type=str, help="The graph name"
    )
    parser.add_argument(
        "--schema", required=True, type=str, help="The schema of the graph"
    )
    parser.add_argument(
        "--num-parts", required=True, type=int, help="The number of partitions"
    )
    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="The output directory of the partitioned results",
    )
    parser.add_argument(
        "--partitions-dir",
        help="directory of the partition-ids for each node type",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="To enable log level for debugging purposes. Available options: \
			  (Critical, Error, Warning, Info, Debug, Notset), default value \
			  is: Info",
    )

    # arguments needed for the distributed implementation
    parser.add_argument(
        "--world-size",
        help="no. of processes to spawn",
        default=1,
        type=int,
        required=True,
    )
    parser.add_argument(
        "--process-group-timeout",
        required=True,
        type=int,
        help="timeout[seconds] for operations executed against the process group "
        "(see torch.distributed.init_process_group)",
    )
    parser.add_argument(
        "--save-orig-nids",
        action="store_true",
        help="Save original node IDs into files",
    )
    parser.add_argument(
        "--save-orig-eids",
        action="store_true",
        help="Save original edge IDs into files",
    )
    parser.add_argument(
        "--use-graphbolt",
        action="store_true",
        help="Use GraphBolt for distributed partition.",
    )
    parser.add_argument(
        "--store-inner-node",
        action="store_true",
        default=False,
        help="Store inner nodes.",
    )

    parser.add_argument(
        "--store-inner-edge",
        action="store_true",
        default=False,
        help="Store inner edges.",
    )
    parser.add_argument(
        "--store-eids",
        action="store_true",
        default=False,
        help="Store edge IDs.",
    )
    parser.add_argument(
        "--graph-formats",
        default=None,
        type=str,
        help="Save partitions in specified formats.",
    )
    params = parser.parse_args()
    # invoke the pipeline function
    numeric_level = getattr(logging, params.log_level.upper(), None)
    logging.basicConfig(
        level=numeric_level,
        format=f"[{platform.node()} %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )
    multi_machine_run(params)
