"""Launching distributed graph partitioning pipeline """
import argparse
import json
import logging
import os
import sys

from partition_algo.base import load_partition_meta

INSTALL_DIR = os.path.abspath(os.path.join(__file__, ".."))
LAUNCH_SCRIPT = "distgraphlaunch.py"
PIPELINE_SCRIPT = "distpartitioning/data_proc_pipeline.py"

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
LARG_SSH_PORT = "ssh_port"


def get_launch_cmd(args) -> str:
    cmd = sys.executable + " " + os.path.join(INSTALL_DIR, LAUNCH_SCRIPT)
    cmd = f"{cmd} --{LARG_SSH_PORT} {args.ssh_port} "
    cmd = f"{cmd} --{LARG_PROCS_MACHINE} 1 "
    cmd = f"{cmd} --{LARG_IPCONF} {args.ip_config} "
    cmd = f"{cmd} --{LARG_MASTER_PORT} {args.master_port} "

    return cmd


def submit_jobs(args) -> str:
    # read the json file and get the remaining argument here.
    schema_path = args.metadata_filename
    with open(os.path.join(args.in_dir, schema_path)) as schema:
        schema_map = json.load(schema)

    graph_name = schema_map["graph_name"]

    # retrieve num_parts
    num_parts = 0
    partition_path = os.path.join(args.partitions_dir, "partition_meta.json")
    if os.path.isfile(partition_path):
        part_meta = load_partition_meta(partition_path)
        num_parts = part_meta.num_parts

    assert (
        num_parts != 0
    ), f"Invalid value for no. of partitions. Please check partition_meta.json file."

    # verify ip_config
    with open(args.ip_config, "r") as f:
        num_ips = len(f.readlines())
        assert (
            num_parts % num_ips == 0
        ), f"The num_parts[{args.num_parts}] should be a multiple of number of lines(ip addresses)[{args.ip_config}]."

    argslist = ""
    argslist += "--world-size {} ".format(num_ips)
    argslist += "--partitions-dir {} ".format(
        os.path.abspath(args.partitions_dir)
    )
    argslist += "--input-dir {} ".format(os.path.abspath(args.in_dir))
    argslist += "--graph-name {} ".format(graph_name)
    argslist += "--schema {} ".format(schema_path)
    argslist += "--num-parts {} ".format(num_parts)
    argslist += "--output {} ".format(os.path.abspath(args.out_dir))
    argslist += "--process-group-timeout {} ".format(args.process_group_timeout)
    argslist += "--log-level {} ".format(args.log_level)
    argslist += "--save-orig-nids " if args.save_orig_nids else ""
    argslist += "--save-orig-eids " if args.save_orig_eids else ""
    argslist += "--use-graphbolt " if args.use_graphbolt else ""
    argslist += "--store-eids " if args.store_eids else ""
    argslist += "--store-inner-node " if args.store_inner_node else ""
    argslist += "--store-inner-edge " if args.store_inner_edge else ""
    argslist += (
        f"--graph-formats {args.graph_formats} " if args.graph_formats else ""
    )

    # (BarclayII) Is it safe to assume all the workers have the Python executable at the same path?
    pipeline_cmd = os.path.join(INSTALL_DIR, PIPELINE_SCRIPT)
    udf_cmd = f"{args.python_path} {pipeline_cmd} {argslist}"

    launch_cmd = get_launch_cmd(args)
    launch_cmd += '"' + udf_cmd + '"'

    print(launch_cmd)
    os.system(launch_cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Dispatch edge index and data to partitions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--in-dir",
        type=str,
        help="Location of the input directory where the dataset is located",
    )
    parser.add_argument(
        "--metadata-filename",
        type=str,
        default="metadata.json",
        help="Filename for the metadata JSON file that describes the dataset to be dispatched.",
    )
    parser.add_argument(
        "--partitions-dir",
        type=str,
        help="Location of the partition-id mapping files which define node-ids and their respective partition-ids, relative to the input directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        help="Location of the output directory where the graph partitions will be created by this pipeline",
    )
    parser.add_argument(
        "--ip-config",
        type=str,
        help="File location of IP configuration for server processes",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=12345,
        help="port used by gloo group to create randezvous point",
    )
    parser.add_argument(
        "--log-level",
        required=False,
        type=str,
        help="Log level to use for execution.",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    parser.add_argument(
        "--python-path",
        type=str,
        default=sys.executable,
        help="Path to the Python executable on all workers",
    )
    parser.add_argument("--ssh-port", type=int, default=22, help="SSH Port.")
    parser.add_argument(
        "--process-group-timeout",
        type=int,
        default=1800,
        help="timeout[seconds] for operations executed against the process group",
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
        type=str,
        default=None,
        help="Save partitions in specified formats. It could be any combination(joined with ``,``) "
        "of ``coo``, ``csc`` and ``csr``. If not specified, save one format only according to "
        "what format is available. If multiple formats are available, selection priority "
        "from high to low is ``coo``, ``csc``, ``csr``.",
    )

    args, _ = parser.parse_known_args()

    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        format=fmt,
        level=getattr(logging, args.log_level, None),
    )

    assert os.path.isdir(args.in_dir)
    assert os.path.isdir(args.partitions_dir)
    assert os.path.isfile(args.ip_config)
    assert isinstance(args.master_port, int)

    submit_jobs(args)


if __name__ == "__main__":
    main()
