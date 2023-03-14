import argparse
import logging
import os
import platform
import sys
from pathlib import Path

import constants

import numpy as np
import pyarrow
import pyarrow.csv as csv
from partition_algo.base import dump_partition_meta, PartitionMeta
from utils import get_idranges, get_node_types, read_json


def post_process(params):
    """Auxiliary function to read the parmetis output file and generate
    metis partition-id files, sorted, per node-type. These files are used
    by the dist. graph partitioning pipeline for further processing.

    Parameters:
    -----------
    params : argparser object
        argparser object to capture command line options passed to the
        executable
    """
    logging.info("Starting to process parmetis output.")

    logging.info(params.postproc_input_dir)
    logging.info(params.schema_file)
    logging.info(params.parmetis_output_file)
    assert os.path.isfile(
        os.path.join(params.postproc_input_dir, params.schema_file)
    )
    assert os.path.isfile(params.parmetis_output_file)
    schema = read_json(
        os.path.join(params.postproc_input_dir, params.schema_file)
    )

    metis_df = csv.read_csv(
        params.parmetis_output_file,
        read_options=pyarrow.csv.ReadOptions(autogenerate_column_names=True),
        parse_options=pyarrow.csv.ParseOptions(delimiter=" "),
    )
    global_nids = metis_df["f0"].to_numpy()
    partition_ids = metis_df["f1"].to_numpy()
    num_parts = np.unique(partition_ids).size

    sort_idx = np.argsort(global_nids)
    global_nids = global_nids[sort_idx]
    partition_ids = partition_ids[sort_idx]

    ntypes_ntypeid_map, ntypes, ntid_ntype_map = get_node_types(schema)
    type_nid_dict, ntype_gnid_offset = get_idranges(
        schema[constants.STR_NODE_TYPE],
        dict(
            zip(
                schema[constants.STR_NODE_TYPE],
                schema[constants.STR_NUM_NODES_PER_TYPE],
            )
        ),
    )

    outdir = Path(params.partitions_dir)
    os.makedirs(outdir, exist_ok=True)
    for ntype_id, ntype_name in ntid_ntype_map.items():
        start = ntype_gnid_offset[ntype_name][0, 0]
        end = ntype_gnid_offset[ntype_name][0, 1]
        out_data = partition_ids[start:end]

        out_file = os.path.join(outdir, f"{ntype_name}.txt")
        options = csv.WriteOptions(include_header=False, delimiter=" ")

        csv.write_csv(
            pyarrow.Table.from_arrays([out_data], names=["partition-ids"]),
            out_file,
            options,
        )
        logging.info(f"Generated {out_file}")

    # generate partition meta file.
    part_meta = PartitionMeta(
        version="1.0.0", num_parts=num_parts, algo_name="metis"
    )
    dump_partition_meta(part_meta, os.path.join(outdir, "partition_meta.json"))

    logging.info("Done processing parmetis output")


if __name__ == "__main__":
    """Main function to convert the output of parmetis into metis partitions
    which are accepted by graph partitioning pipeline.

    ParMETIS currently generates one output file, which is in the following format:
    <global-node-id> <partition-id>

    Graph partitioing pipeline, per the new dataset file format rules expects the
    metis partitions to be in the following format:
    No. of files will be equal to the no. of node-types in the graph
    Each file will have one-number/line which is <partition-id>.

    Example usage:
    --------------
    python parmetis_postprocess.py
        --input_file <metis-partitions-file>
        --output-dir <directory where the output files are stored>
        --schema <schema-file-path>
    """
    parser = argparse.ArgumentParser(
        description="PostProcessing the ParMETIS\
        output for partitioning pipeline"
    )
    parser.add_argument(
        "--postproc_input_dir",
        required=True,
        type=str,
        help="Base directory for post processing step.",
    )
    parser.add_argument(
        "--schema_file",
        required=True,
        type=str,
        help="The schema of the input graph",
    )
    parser.add_argument(
        "--parmetis_output_file",
        required=True,
        type=str,
        help="ParMETIS output file",
    )
    parser.add_argument(
        "--partitions_dir",
        required=True,
        type=str,
        help="The output\
        will be files (with metis partition ids) and each file corresponds to\
        a node-type in the input graph dataset.",
    )
    params = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level="INFO",
        format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )

    # Invoke the function for post processing
    post_process(params)
