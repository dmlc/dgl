import argparse
import json
import logging
import os
import time

import dgl

import torch
from dgl._ffi.base import DGLError
from dgl.data.utils import load_graphs
from dgl.utils import toindex

ETYPES_KEY = "etypes"
EDGE_MAP_KEY = "edge_map"
NTYPES_KEY = "ntypes"
NUM_PARTS_KEY = "num_parts"
CANONICAL_ETYPE_DELIMITER = ":"


def convert_conf(part_config):
    with open(part_config, "r+", encoding="utf-8") as f:
        config = json.load(f)
        logging.info("Checking if the provided json file need to be changed.")
        if is_old_version(config):
            logging.info("Changing the partition configuration file.")
            canonical_etypes = {}
            if len(config[NTYPES_KEY]) == 1:
                ntype = list(config[NTYPES_KEY].keys())[0]
                canonical_etypes = {
                    CANONICAL_ETYPE_DELIMITER.join((ntype, etype, ntype)): eid
                    for etype, eid in config[ETYPES_KEY].items()
                }
            else:
                canonical_etypes = etype2canonical_etype(part_config, config)
            reverse_c_etypes = {v: k for k, v in canonical_etypes.items()}
            # Convert edge_map keys from etype -> c_etype.
            new_edge_map = {}
            for e_type, range in config[EDGE_MAP_KEY].items():
                eid = config[ETYPES_KEY][e_type]
                c_etype = reverse_c_etypes[eid]
                new_edge_map[c_etype] = range
            config[EDGE_MAP_KEY] = new_edge_map
            config[ETYPES_KEY] = canonical_etypes
            logging.info("Dumping the content to disk.")
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()


def etype2canonical_etype(part_config, config):
    num_parts = config[NUM_PARTS_KEY]
    edge_map = config[EDGE_MAP_KEY]
    etypes = list(edge_map.keys())
    # Get part id of each seed edge.
    partition_ids = []
    for _, bound in edge_map.items():
        for i in range(num_parts):
            if bound[i][1] > bound[i][0]:
                partition_ids.append(i)
                break
    partition_ids = torch.tensor(partition_ids)

    # Get starting index of each partition.
    shifts = []
    for i in range(num_parts):
        shifts.append(edge_map[etypes[0]][i][0])
    shifts = torch.tensor(shifts)

    canonical_etypes = {}
    part_ids = [
        part_id for part_id in range(num_parts) if part_id in partition_ids
    ]
    for part_id in part_ids:
        seed_etypes = [
            etypes[i] for i in range(len(etypes)) if partition_ids[i] == part_id
        ]
        c_etype = _find_c_etypes_in_partition(
            part_id,
            seed_etypes,
            config[ETYPES_KEY],
            config[NTYPES_KEY],
            edge_map,
            shifts,
            part_config,
        )
        canonical_etypes.update(c_etype)
    return canonical_etypes


def _find_c_etypes_in_partition(
    part_id, seed_etypes, etypes, ntypes, edge_map, shifts, config_path
):
    try:
        folder = os.path.dirname(os.path.realpath(config_path))
        local_g = load_graphs(f"{folder}/part{part_id}/graph.dgl")[0][0]
        local_eids = [
            edge_map[etype][part_id][0] - shifts[part_id]
            for etype in seed_etypes
        ]
        local_eids = toindex(torch.tensor(local_eids))
        local_eids = local_eids.tousertensor()
        local_src, local_dst = local_g.find_edges(local_eids)
        src_ntids, dst_ntids = (
            local_g.ndata[dgl.NTYPE][local_src],
            local_g.ndata[dgl.NTYPE][local_dst],
        )
        ntypes = {v: k for k, v in ntypes.items()}
        src_ntypes = [ntypes[ntid.item()] for ntid in src_ntids]
        dst_ntypes = [ntypes[ntid.item()] for ntid in dst_ntids]
        c_etypes = list(zip(src_ntypes, seed_etypes, dst_ntypes))
        c_etypes = [
            CANONICAL_ETYPE_DELIMITER.join(c_etype) for c_etype in c_etypes
        ]
        return {k: etypes[v] for (k, v) in zip(c_etypes, seed_etypes)}
    except DGLError as e:
        print(e)
        logging.fatal(
            f"Graph data of partition {part_id} is requested but not found."
        )


def is_old_version(config):
    first_etype = list(config[ETYPES_KEY].keys())[0]
    etype_tuple = first_etype.split(CANONICAL_ETYPE_DELIMITER)
    return len(etype_tuple) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change edge type in config file from format (str)"
        " to (str,str,str), the original file will be overwritten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--part_config", type=str, help="The file of the partition config"
    )
    args, _ = parser.parse_known_args()
    assert (
        args.part_config is not None
    ), "A user has to specify a partition config file with --part_config."

    start = time.time()
    convert_conf(args.part_config)
    end = time.time()
    logging.info(f"elplased time in seconds: {end - start}")
