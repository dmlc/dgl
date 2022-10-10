"""Copy the partitions to a cluster of machines."""
import argparse
import copy
import json
import logging
import os
import signal
import stat
import subprocess
import sys


def copy_file(file_name, ip, workspace, param=""):
    print("copy {} to {}".format(file_name, ip + ":" + workspace + "/"))
    cmd = "scp " + param + " " + file_name + " " + ip + ":" + workspace + "/"
    subprocess.check_call(cmd, shell=True)


def exec_cmd(ip, cmd):
    cmd = "ssh -o StrictHostKeyChecking=no " + ip + " '" + cmd + "'"
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Copy data to the servers.")
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Path of user directory of distributed tasks. \
                        This is used to specify a destination location where \
                        data are copied to on remote machines.",
    )
    parser.add_argument(
        "--rel_data_path",
        type=str,
        required=True,
        help="Relative path in workspace to store the partition data.",
    )
    parser.add_argument(
        "--part_config",
        type=str,
        required=True,
        help="The partition config file. The path is on the local machine.",
    )
    parser.add_argument(
        "--script_folder",
        type=str,
        required=True,
        help="The folder contains all the user code scripts.",
    )
    parser.add_argument(
        "--ip_config",
        type=str,
        required=True,
        help="The file of IP configuration for servers. \
                        The path is on the local machine.",
    )
    args = parser.parse_args()

    hosts = []
    with open(args.ip_config) as f:
        for line in f:
            res = line.strip().split(" ")
            ip = res[0]
            hosts.append(ip)

    # We need to update the partition config file so that the paths are relative to
    # the workspace in the remote machines.
    with open(args.part_config) as conf_f:
        part_metadata = json.load(conf_f)
        tmp_part_metadata = copy.deepcopy(part_metadata)
        num_parts = part_metadata["num_parts"]
        assert num_parts == len(
            hosts
        ), "The number of partitions needs to be the same as the number of hosts."
        graph_name = part_metadata["graph_name"]
        node_map = part_metadata["node_map"]
        edge_map = part_metadata["edge_map"]
        if not isinstance(node_map, dict):
            assert (
                node_map[-4:] == ".npy"
            ), "node map should be stored in a NumPy array."
            tmp_part_metadata["node_map"] = "{}/{}/node_map.npy".format(
                args.workspace, args.rel_data_path
            )
        if not isinstance(edge_map, dict):
            assert (
                edge_map[-4:] == ".npy"
            ), "edge map should be stored in a NumPy array."
            tmp_part_metadata["edge_map"] = "{}/{}/edge_map.npy".format(
                args.workspace, args.rel_data_path
            )

        for part_id in range(num_parts):
            part_files = tmp_part_metadata["part-{}".format(part_id)]
            part_files["edge_feats"] = "{}/part{}/edge_feat.dgl".format(
                args.rel_data_path, part_id
            )
            part_files["node_feats"] = "{}/part{}/node_feat.dgl".format(
                args.rel_data_path, part_id
            )
            part_files["part_graph"] = "{}/part{}/graph.dgl".format(
                args.rel_data_path, part_id
            )
    tmp_part_config = "/tmp/{}.json".format(graph_name)
    with open(tmp_part_config, "w") as outfile:
        json.dump(tmp_part_metadata, outfile, sort_keys=True, indent=4)

    # Copy ip config.
    for part_id, ip in enumerate(hosts):
        remote_path = "{}/{}".format(args.workspace, args.rel_data_path)
        exec_cmd(ip, "mkdir -p {}".format(remote_path))

        copy_file(args.ip_config, ip, args.workspace)
        copy_file(
            tmp_part_config,
            ip,
            "{}/{}".format(args.workspace, args.rel_data_path),
        )
        node_map = part_metadata["node_map"]
        edge_map = part_metadata["edge_map"]
        if not isinstance(node_map, dict):
            copy_file(node_map, ip, tmp_part_metadata["node_map"])
        if not isinstance(edge_map, dict):
            copy_file(edge_map, ip, tmp_part_metadata["edge_map"])
        remote_path = "{}/{}/part{}".format(
            args.workspace, args.rel_data_path, part_id
        )
        exec_cmd(ip, "mkdir -p {}".format(remote_path))

        part_files = part_metadata["part-{}".format(part_id)]
        copy_file(part_files["node_feats"], ip, remote_path)
        copy_file(part_files["edge_feats"], ip, remote_path)
        copy_file(part_files["part_graph"], ip, remote_path)
        # copy script folder
        copy_file(args.script_folder, ip, args.workspace, "-r")


def signal_handler(signal, frame):
    logging.info("Stop copying")
    sys.exit(0)


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO)
    signal.signal(signal.SIGINT, signal_handler)
    main()
