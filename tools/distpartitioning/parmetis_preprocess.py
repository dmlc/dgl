import argparse
import logging
import os
import platform
from pathlib import Path

import array_readwriter

import constants

import numpy as np
import pyarrow
import pyarrow.csv as csv
from utils import (
    generate_read_list,
    generate_roundrobin_read_list,
    get_idranges,
    get_node_types,
    read_json,
)


def get_proc_info():
    """Helper function to get the rank from the
    environment when `mpirun` is used to run this python program.

    Please note that for mpi(openmpi) installation the rank is retrieved from the
    environment using OMPI_COMM_WORLD_RANK. For mpich it is
    retrieved from the environment using PMI_RANK.

    Returns:
    --------
    integer :
        Rank of the current process.
    """
    env_variables = dict(os.environ)
    # mpich
    if "PMI_RANK" in env_variables:
        return int(env_variables["PMI_RANK"])
    # openmpi
    elif "OMPI_COMM_WORLD_RANK" in env_variables:
        return int(env_variables["OMPI_COMM_WORLD_RANK"])
    else:
        return 0


def get_world_size():
    """Helper function to get the world size from the
    environment when `mpirun` is used to run this python program.

    Returns:
    --------
    integer :
        Numer of processes created by the executor that created this process.
    """
    env_variables = dict(os.environ)
    # mpich
    if "PMI_SIZE" in env_variables:
        return int(env_variables["PMI_SIZE"])
    # openmpi
    elif "OMPI_COMM_WORLD_SIZE" in env_variables:
        return int(env_variables["OMPI_COMM_WORLD_SIZE"])
    else:
        return 1


def gen_edge_files(rank, schema_map, params):
    """Function to create edges files to be consumed by ParMETIS
    for partitioning purposes.

    This function creates the edge files and each of these will have the
    following format (meaning each line of these file is of the following format)
    <global_src_id> <global_dst_id>

    Here ``global`` prefix means that globally unique identifier assigned each node
    in the input graph. In this context globally unique means unique across all the
    nodes in the input graph.

    Parameters:
    -----------
    rank : int
        rank of the current process
    schema_map : json dictionary
        Dictionary created by reading the metadata.json file for the input dataset.
    output : string
        Location of storing the node-weights and edge files for ParMETIS.
    """
    _, ntype_gnid_offset = get_idranges(
        schema_map[constants.STR_NODE_TYPE],
        dict(
            zip(
                schema_map[constants.STR_NODE_TYPE],
                schema_map[constants.STR_NUM_NODES_PER_TYPE],
            )
        ),
    )

    # Regenerate edge files here.
    edge_data = schema_map[constants.STR_EDGES]

    outdir = Path(params.output_dir)
    os.makedirs(outdir, exist_ok=True)

    def process_and_write_back(data_df, idx):
        data_f0 = data_df[:, 0]
        data_f1 = data_df[:, 1]

        global_src_id = data_f0 + ntype_gnid_offset[src_ntype_name][0, 0]
        global_dst_id = data_f1 + ntype_gnid_offset[dst_ntype_name][0, 0]
        cols = [global_src_id, global_dst_id]
        col_names = ["global_src_id", "global_dst_id"]

        out_file_name = Path(edge_data_files[idx]).stem.split(".")[0]
        out_file = os.path.join(
            outdir, etype_name, f"edges_{out_file_name}.csv"
        )
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        options = csv.WriteOptions(include_header=False, delimiter=" ")
        csv.write_csv(
            pyarrow.Table.from_arrays(cols, names=col_names),
            out_file,
            options,
        )
        return out_file

    edge_files = []
    for etype_name, etype_info in edge_data.items():
        edge_data_files = etype_info[constants.STR_DATA]

        # ``edgetype`` strings are in canonical format, src_node_type:edge_type:dst_node_type
        tokens = etype_name.split(":")
        assert len(tokens) == 3

        src_ntype_name = tokens[0]

        dst_ntype_name = tokens[2]

        rank_assignments = generate_roundrobin_read_list(
            len(edge_data_files), params.num_parts
        )
        for file_idx in rank_assignments[rank]:
            reader_fmt_meta = {
                "name": etype_info[constants.STR_FORMAT][constants.STR_NAME],
            }
            if reader_fmt_meta["name"] == constants.STR_CSV:
                reader_fmt_meta["delimiter"] = etype_info[constants.STR_FORMAT][
                    constants.STR_FORMAT_DELIMITER
                ]
            data_df = array_readwriter.get_array_parser(**reader_fmt_meta).read(
                os.path.join(params.input_dir, edge_data_files[file_idx])
            )
            out_file = process_and_write_back(data_df, file_idx)
            edge_files.append(out_file)

    return edge_files


def gen_node_weights_files(schema_map, params):
    """Function to create node weight files for ParMETIS along with the edge files.

    This function generates node-data files, which will be read by the ParMETIS
    executable for partitioning purposes. Each line in these files will be of the
    following format:
        <node_type_id> <node_weight_list> <type_wise_node_id>
    node_type_id -  is id assigned to the node-type to which a given particular
        node belongs to
    weight_list - this is a one-hot vector in which the number in the location of
        the current nodes' node-type will be set to `1` and other will be `0`
    type_node_id - this is the id assigned to the node (in the context of the current
        nodes` node-type). Meaning this id is unique across all the nodes which belong to
        the current nodes` node-type.

    Parameters:
    -----------
    schema_map : json dictionary
        Dictionary created by reading the metadata.json file for the input dataset.
    output : string
        Location of storing the node-weights and edge files for ParMETIS.

    Returns:
    --------
    list :
        List of filenames for nodes of the input graph.
    list :
        List o ffilenames for edges of the input graph.
    """
    rank = get_proc_info()
    ntypes_ntypeid_map, ntypes, ntid_ntype_map = get_node_types(schema_map)
    type_nid_dict, ntype_gnid_offset = get_idranges(
        schema_map[constants.STR_NODE_TYPE],
        dict(
            zip(
                schema_map[constants.STR_NODE_TYPE],
                schema_map[constants.STR_NUM_NODES_PER_TYPE],
            )
        ),
    )

    node_files = []
    outdir = Path(params.output_dir)
    os.makedirs(outdir, exist_ok=True)

    for ntype_id, ntype_name in ntid_ntype_map.items():

        # This ntype does not have any train/test/val masks...
        # Each rank will generate equal no. of rows for this node type.
        total_count = schema_map[constants.STR_NUM_NODES_PER_TYPE][ntype_id]
        per_rank_range = np.ones((params.num_parts,), dtype=np.int64) * (
            total_count // params.num_parts
        )
        for i in range(total_count % params.num_parts):
            per_rank_range[i] += 1

        tid_start = np.cumsum([0] + list(per_rank_range[:-1]))
        tid_end = np.cumsum(list(per_rank_range))
        local_tid_start = tid_start[rank]
        local_tid_end = tid_end[rank]
        sz = local_tid_end - local_tid_start

        cols = []
        col_names = []

        # ntype-id
        cols.append(
            pyarrow.array(np.ones(sz, dtype=np.int64) * np.int64(ntype_id))
        )
        col_names.append("ntype")

        # one-hot vector for ntype-id here.
        for i in range(len(ntypes)):
            if i == ntype_id:
                cols.append(pyarrow.array(np.ones(sz, dtype=np.int64)))
            else:
                cols.append(pyarrow.array(np.zeros(sz, dtype=np.int64)))
            col_names.append("w{}".format(i))

        # `type_nid` should be the very last column in the node weights files.
        cols.append(
            pyarrow.array(
                np.arange(local_tid_start, local_tid_end, dtype=np.int64)
            )
        )
        col_names.append("type_nid")

        out_file = os.path.join(
            outdir, "node_weights_{}_{}.txt".format(ntype_name, rank)
        )
        options = csv.WriteOptions(include_header=False, delimiter=" ")
        options.delimiter = " "

        csv.write_csv(
            pyarrow.Table.from_arrays(cols, names=col_names), out_file, options
        )
        node_files.append(
            (
                ntype_gnid_offset[ntype_name][0, 0] + local_tid_start,
                ntype_gnid_offset[ntype_name][0, 0] + local_tid_end,
                out_file,
            )
        )

    return node_files


def gen_parmetis_input_args(params, schema_map):
    """Function to create two input arguments which will be passed to the parmetis.
    first argument is a text file which has a list of node-weights files,
    namely parmetis-nfiles.txt, and second argument is a text file which has a
    list of edge files, namely parmetis_efiles.txt.
    ParMETIS uses these two files to read/load the graph and partition the graph
    With regards to the file format, parmetis_nfiles.txt uses the following format
    for each line in that file:
        <filename> <global_node_id_start> <global_node_id_end>(exclusive)
    While parmetis_efiles.txt just has <filename> in each line.

    Parameters:
    -----------
    params : argparser instance
        Instance of ArgParser class, which has all the input arguments passed to
        run this program.
    schema_map : json dictionary
        Dictionary object created after reading the graph metadata.json file.
    """

    # TODO: This makes the assumption that all node files have the same number of chunks
    ntypes_ntypeid_map, ntypes, ntid_ntype_map = get_node_types(schema_map)
    type_nid_dict, ntype_gnid_offset = get_idranges(
        schema_map[constants.STR_NODE_TYPE],
        dict(
            zip(
                schema_map[constants.STR_NODE_TYPE],
                schema_map[constants.STR_NUM_NODES_PER_TYPE],
            )
        ),
    )

    # Check if <graph-name>_stats.txt exists, if not create one using metadata.
    # Here stats file will be created in the current directory.
    # No. of constraints, third column in the stats file is computed as follows:
    #   num_constraints = no. of node types + train_mask + test_mask + val_mask
    #   Here, (train/test/val) masks will be set to 1 if these masks exist for
    #   all the node types in the graph, otherwise these flags will be set to 0
    assert (
        constants.STR_GRAPH_NAME in schema_map
    ), "Graph name is not present in the json file"
    graph_name = schema_map[constants.STR_GRAPH_NAME]
    if not os.path.isfile(
        os.path.join(params.input_dir, f"{graph_name}_stats.txt")
    ):
        num_nodes = np.sum(schema_map[constants.STR_NUM_NODES_PER_TYPE])
        num_edges = np.sum(schema_map[constants.STR_NUM_EDGES_PER_TYPE])
        num_ntypes = len(schema_map[constants.STR_NODE_TYPE])

        num_constraints = num_ntypes

        with open(
            os.path.join(params.input_dir, f"{graph_name}_stats.txt"), "w"
        ) as sf:
            sf.write(f"{num_nodes} {num_edges} {num_constraints}")

    node_files = []
    outdir = Path(params.output_dir)
    os.makedirs(outdir, exist_ok=True)
    for ntype_id, ntype_name in ntid_ntype_map.items():
        global_nid_offset = ntype_gnid_offset[ntype_name][0, 0]
        total_count = schema_map[constants.STR_NUM_NODES_PER_TYPE][ntype_id]
        per_rank_range = np.ones((params.num_parts,), dtype=np.int64) * (
            total_count // params.num_parts
        )
        for i in range(total_count % params.num_parts):
            per_rank_range[i] += 1
        tid_start = np.cumsum([0] + list(per_rank_range[:-1]))
        tid_end = np.cumsum(per_rank_range)
        logging.info(f" tid-start = {tid_start}, tid-end = {tid_end}")
        logging.info(f" per_rank_range - {per_rank_range}")

        for part_idx in range(params.num_parts):
            local_tid_start = tid_start[part_idx]
            local_tid_end = tid_end[part_idx]
            out_file = os.path.join(
                outdir, "node_weights_{}_{}.txt".format(ntype_name, part_idx)
            )
            node_files.append(
                (
                    out_file,
                    global_nid_offset + local_tid_start,
                    global_nid_offset + local_tid_end,
                )
            )

    with open(
        os.path.join(params.output_dir, "parmetis_nfiles.txt"), "w"
    ) as parmetis_nf:
        for node_file in node_files:
            # format: filename global_node_id_start global_node_id_end(exclusive)
            parmetis_nf.write(
                "{} {} {}\n".format(node_file[0], node_file[1], node_file[2])
            )

    # Regenerate edge files here.
    # NOTE: The file names need to match the ones generated by gen_edge_files function
    edge_data = schema_map[constants.STR_EDGES]
    edge_files = []
    for etype_name, etype_info in edge_data.items():
        edge_data_files = etype_info[constants.STR_DATA]
        for edge_file_path in edge_data_files:
            out_file_name = Path(edge_file_path).stem.split(".")[0]
            out_file = os.path.join(
                outdir, etype_name, "edges_{}.csv".format(out_file_name)
            )
            edge_files.append(out_file)

    with open(
        os.path.join(params.output_dir, "parmetis_efiles.txt"), "w"
    ) as parmetis_efile:
        for edge_file in edge_files:
            parmetis_efile.write("{}\n".format(edge_file))


def run_preprocess_data(params):
    """Main function which will help create graph files for ParMETIS processing

    Parameters:
    -----------
    params : argparser object
        An instance of argparser class which stores command line arguments.
    """
    logging.info("Starting to generate ParMETIS files...")
    rank = get_proc_info()

    assert os.path.isdir(
        params.input_dir
    ), f"Please check `input_dir` argument: {params.input_dit}."

    schema_map = read_json(os.path.join(params.input_dir, params.schema_file))
    gen_node_weights_files(schema_map, params)
    logging.info("Done with node weights....")

    gen_edge_files(rank, schema_map, params)
    logging.info("Done with edge weights...")

    if rank == 0:
        gen_parmetis_input_args(params, schema_map)
    logging.info("Done generating files for ParMETIS run ..")


if __name__ == "__main__":
    """Main function used to generate temporary files needed for ParMETIS execution.
    This function generates node-weight files and edges files which are consumed by ParMETIS.

    Example usage:
    --------------
    mpirun -np 4 python3 parmetis_preprocess.py --schema <file> --output <target-output-dir>
    """
    parser = argparse.ArgumentParser(
        description="Generate ParMETIS files for input dataset"
    )
    parser.add_argument(
        "--schema_file",
        required=True,
        type=str,
        help="The schema of the input graph",
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        type=str,
        help="This directory will be used as the relative directory to locate files, if absolute paths are not used",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="The output directory for the node weights files and auxiliary files for ParMETIS.",
    )
    parser.add_argument(
        "--num_parts",
        required=True,
        type=int,
        help="Total no. of output graph partitions.",
    )
    parser.add_argument(
        "--log_level",
        required=False,
        type=str,
        help="Log level to use for execution.",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    params = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level=getattr(logging, params.log_level, None),
        format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )

    # Invoke the function to generate files for parmetis
    run_preprocess_data(params)
