import argparse
import logging
import os
import platform
import sys
from pathlib import Path

import constants
from utils import read_json


def check_dependencies():
    """Check if all the dependencies needed for the execution of this file
    are installed.
    """

    exec_path = os.get_exec_path()
    mpi_install = False
    for x in exec_path:
        if os.path.isfile(os.path.join(x, "mpirun")):
            mpi_install = True
            break
    assert (
        mpi_install
    ), "Could not locate the following dependency: MPI. Please install it and try again."

    dgl_path = os.environ.get("DGL_HOME", "")
    assert os.path.isdir(
        dgl_path
    ), "Environment variable DGL_HOME not found. Please define the DGL installation path"


def run_parmetis_wrapper(params):
    """Function to execute all the steps needed to run ParMETIS

    Parameters:
    -----------
    params : argparser object
        an instance of argparser class to capture command-line arguments
    """
    schema = read_json(
        os.path.join(params.preproc_input_dir, params.schema_file)
    )
    graph_name = schema[constants.STR_GRAPH_NAME]
    num_partitions = params.num_parts

    # Check if parmetis_preprocess.py exists.
    assert os.path.isfile(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "parmetis_preprocess.py"
        )
    ), "Please check DGL Installation, parmetis_preprocess.py file does not exist."

    # Trigger pre-processing step to generate input files for ParMETIS.
    preproc_cmd = (
        f"mpirun -np {num_partitions} -hostfile {params.hostfile} "
        f"python3 $DGL_HOME/tools/distpartitioning/parmetis_preprocess.py "
        f"--schema_file {params.schema_file} "
        f"--input_dir {params.preproc_input_dir} "
        f"--output_dir {params.preproc_output_dir} "
        f"--num_parts {num_partitions}"
    )
    logging.info(f"Executing Preprocessing Step: {preproc_cmd}")
    os.system(preproc_cmd)
    logging.info(f"Done Preprocessing Step")

    # Trigger ParMETIS for creating metis partitions for the input graph.
    parmetis_install_path = "pm_dglpart3"
    if params.parmetis_install_path is not None:
        parmetis_install_path = os.path.join(
            params.parmetis_install_path, parmetis_install_path
        )
    parmetis_nfiles = os.path.join(
        params.preproc_output_dir, "parmetis_nfiles.txt"
    )
    parmetis_efiles = os.path.join(
        params.preproc_output_dir, "parmetis_efiles.txt"
    )
    parmetis_cmd = (
        f"mpirun -np {num_partitions} -hostfile {params.hostfile} "
        f"{parmetis_install_path} {graph_name} {num_partitions} "
        f"{parmetis_nfiles} {parmetis_efiles}"
    )
    logging.info(f"Executing ParMETIS: {parmetis_cmd}")
    os.system(parmetis_cmd)
    logging.info(f"Done ParMETIS execution step")

    # Trigger post-processing step to convert parmetis output to the form
    # acceptable by dist. graph partitioning pipeline.
    parmetis_output_file = os.path.join(
        os.getcwd(), f"{graph_name}_part.{num_partitions}"
    )
    postproc_cmd = (
        f"python3 $DGL_HOME/tools/distpartitioning/parmetis_postprocess.py "
        f"--postproc_input_dir {params.preproc_input_dir} "
        f"--schema_file {params.schema_file} "
        f"--parmetis_output_file {parmetis_output_file} "
        f"--partitions_dir {params.partitions_dir}"
    )
    logging.info(f"Executing PostProcessing: {postproc_cmd}")
    os.system(postproc_cmd)
    logging.info("Done Executing ParMETIS...")


if __name__ == "__main__":
    """Main function to invoke the parmetis wrapper function"""
    parser = argparse.ArgumentParser(
        description="Run ParMETIS as part of the graph partitioning pipeline"
    )
    # Preprocessing step.
    parser.add_argument(
        "--schema_file",
        required=True,
        type=str,
        help="The schema of the input graph",
    )
    parser.add_argument(
        "--preproc_input_dir",
        type=str,
        help="The input directory for preprocess where the dataset is located",
    )
    parser.add_argument(
        "--preproc_output_dir",
        required=True,
        type=str,
        help="The output directory for the node weights files and auxiliary\
              files for ParMETIS.",
    )
    parser.add_argument(
        "--hostfile",
        required=True,
        type=str,
        help="A text file with a list of ip addresses.",
    )
    parser.add_argument(
        "--num_parts",
        required=True,
        type=int,
        help="integer representing no. of partitions.",
    )

    # ParMETIS step.
    parser.add_argument(
        "--parmetis_install_path",
        required=False,
        type=str,
        help="The directory where ParMETIS is installed",
    )

    # Postprocessing step.
    parser.add_argument(
        "--parmetis_output_file",
        required=True,
        type=str,
        help="ParMETIS output file (global_node_id to partition_id mappings)",
    )
    parser.add_argument(
        "--partitions_dir",
        required=True,
        type=str,
        help="The directory where the files (with metis partition ids) grouped \
              by node_types",
    )
    params = parser.parse_args()

    # Configure logging.
    logging.basicConfig(
        level="INFO",
        format=f"[{platform.node()} \
        %(levelname)s %(asctime)s PID:%(process)d] %(message)s",
    )

    check_dependencies()
    run_parmetis_wrapper(params)
