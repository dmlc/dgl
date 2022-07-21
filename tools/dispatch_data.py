"""Launching distributed graph partitioning pipeline """
import os
import sys
import argparse
import logging
import json

INSTALL_DIR = "/home/ubuntu/workspace/DistDgl-Utils-updatedfileformat"
LAUNCH_SCRIPT = "distgraphlaunch.py"
PIPELINE_SCRIPT = "data_proc_pipeline.py"

UDF_WORLD_SIZE = "world-size"
UDF_PART_DIR = "partitions-dir"
UDF_INPUT_DIR = "input-dir"
UDF_GRAPH_NAME = "graph-name"
UDF_SCHEMA = "schema"
UDF_NUM_PARTS = "num-parts"
UDF_OUT_DIR = "output"

LARG_PROCS_MACHINE = "num_proc_per_machine"
LARG_IPCONF = "ip_config"
LARG_MASTER_PORT = "master_port"

def get_launch_cmd(args, user_python_exe) -> str: 

    cmd = pythonpath=os.environ.get("PYTHONPATH", "") + get_python_path(user_python_exe) + " " + os.path.join(INSTALL_DIR, LAUNCH_SCRIPT)
    cmd = f"{cmd} --{LARG_PROCS_MACHINE} 1 "
    cmd = f"{cmd} --{LARG_IPCONF} {args.ip_config} "
    cmd = f"{cmd} --{LARG_MASTER_PORT} {args.master_port} "

    return cmd

def get_python_path(user_python_exe) -> str:

    # Auto-detect the python binary that kicks off the distributed trainer code.
    # Note: This allowlist order matters, this will match with the FIRST matching entry. Thus, please add names to this
    #       from most-specific to least-specific order eg:
    #           (python3.7, python3.8) -> (python3)
    # The allowed python versions are from this: https://www.dgl.ai/pages/start.html
    python_bin_allowlist = (
        # for backwards compatibility, accept python2 but technically DGL is a py3 library, so this is not recommended
        "python2.7", "python2",
        "python3.6", "python3.7", "python3.8", "python3.9", "python3",
    )

    # If none of the candidate python bins match, then we go with the default `python`
    python_bin = "python"
    for candidate_python_bin in python_bin_allowlist:
        if candidate_python_bin == user_python_exe :
            python_bin = candidate_python_bin
            break
    return python_bin


def submit_jobs(args, user_python_exe) -> str:
    wrapper_command = os.path.join(INSTALL_DIR, LAUNCH_SCRIPT)

    UDF_ARGS = [UDF_WORLD_SIZE, UDF_PART_DIR, UDF_INPUT_DIR, UDF_GRAPH_NAME, UDF_SCHEMA, UDF_NUM_PARTS, UDF_OUT_DIR]

    #read the json file and get the remaining argument here. 
    with open(os.path.join(args.in_dir, "meta.json")) as schema:
        schema_map = json.load(schema)

    num_parts = len(schema_map["num_nodes_per_chunk"][0])
    graph_name = schema_map["graph_name"]

    argslist = ""
    for cmd_arg in UDF_ARGS: 
        if cmd_arg == "world-size": 
            argslist += "--world-size 4 ".format(num_parts)
        elif cmd_arg == "partitions-dir":
            argslist += "--partitions-dir {} ".format(args.partitions_dir)
        elif cmd_arg == "input-dir":
            argslist += "--input-dir {} ".format(args.in_dir)
        elif cmd_arg == "graph-name":
            argslist += "--graph-name {} ".format(graph_name)
        elif cmd_arg == "schema":
            argslist += "--schema metadata.json "
        elif cmd_arg == "num-parts":
            argslist += "--num-parts {} ".format(num_parts)
        elif cmd_arg == "output":
            argslist += "--output {} ".format(args.out_dir)
        
    pythonpath=os.environ.get("PYTHONPATH", "")+get_python_path(user_python_exe)
    pipeline_cmd = os.path.join(INSTALL_DIR, PIPELINE_SCRIPT)
    udf_cmd = f"{pythonpath} {pipeline_cmd} {argslist}"

    launch_cmd = get_launch_cmd(args, user_python_exe)
    launch_cmd += '\"'+udf_cmd+'\"'

    print(launch_cmd)
    os.system(launch_cmd)

def main():
    parser = argparse.ArgumentParser(description='Launch a distributed job')

    parser.add_argument('--in-dir', type=str, help='Location of the input directory where the dataset is located')
    parser.add_argument('--partitions-dir', type=str, help='Location of the partition-id mapping files which define node-ids and their respective partition-ids')
    parser.add_argument('--out-dir', type=str, help='Location of the output directory where the graph partitions will be created by this pipeline')
    parser.add_argument('--ip-config', type=str, help='File location of IP configuration for server processes')
    parser.add_argument('--master-port', type=int, help='port used by gloo group to create randezvous point')

    args, udf_command = parser.parse_known_args()

    assert os.path.isdir(args.in_dir)
    assert os.path.isdir(args.partitions_dir)
    assert os.path.isfile(args.ip_config)
    assert isinstance(args.master_port, int)

    tokens = sys.executable.split(os.sep)
    submit_jobs(args, tokens[-1])

if __name__ == '__main__':
    fmt = '%(asctime)s %(levelname)s %(message)s'
    logging.basicConfig(format=fmt, level=logging.INFO)
    main()
