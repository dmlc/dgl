import argparse
import json
import logging
import os
import time

import torch

import dgl
from dgl._ffi.base import DGLError
from dgl.data.utils import load_graphs
from dgl.distributed import load_partition_book

etypes_key = "etypes"
edge_map_key = "edge_map"
canonical_etypes_delimiter = ":"


def convert_conf(part_config):
    with open(part_config, "r+", encoding="utf-8") as f:
        config = json.load(f)
        logging.info("Checking if the provided json file need to be changed.")
        if is_old_version(config):
            logging.info("Changing the partition configuration file.")
            canonical_etypes = etype2canonical_etype(part_config)
            # convert edge_map key from etype -> c_etype
            new_edge_map = {}
            for e_type, range in config[edge_map_key].items():
                eid = config[etypes_key][e_type]
                c_etype = [
                    key
                    for key in canonical_etypes
                    if canonical_etypes[key] == eid
                ][0]
                new_edge_map[c_etype] = range
            config[edge_map_key] = new_edge_map
            config[etypes_key] = canonical_etypes
            logging.info("Dumping the content to disk.")
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()


def etype2canonical_etype(part_config):
    gpb, _, _, etypes = load_partition_book(part_config=part_config, part_id=0)
    eid = []
    etype_id = []
    for etype in etypes:
        type_eid = torch.zeros((1,), dtype=torch.int64)
        eid.append(gpb.map_to_homo_eid(type_eid, etype))
        etype_id.append(etypes[etype])
    eid = torch.cat(eid, 0)
    etype_id = torch.IntTensor(etype_id)
    partition_id = gpb.eid2partid(eid)
    canonical_etypes = {}
    part_ids = [
        part_id
        for part_id in range(gpb.num_partitions())
        if part_id in partition_id
    ]
    for part_id in part_ids:
        seed_edges = torch.masked_select(eid, partition_id == part_id)
        seed_edge_tids = torch.masked_select(etype_id, partition_id == part_id)
        c_etype = _find_c_etypes_in_partition(
            seed_edges, seed_edge_tids, part_id, part_config
        )
        canonical_etypes.update(c_etype)
    return canonical_etypes


def _find_c_etypes_in_partition(
    seed_edges, seed_edge_tids, part_id, part_config
):
    folder = os.path.dirname(os.path.realpath(part_config))
    partition_book = {}
    local_g = dgl.DGLGraph()
    try:
        local_g = load_graphs(f"{folder}/part{part_id}/graph.dgl")[0][0]
        partition_book = load_partition_book(
            part_config=part_config, part_id=part_id
        )[0]
    except DGLError as e:
        logging.fatal(
            f"Graph data of partition {part_id} is requested but not found."
        )
        raise e

    ntypes, etypes = partition_book.ntypes, partition_book.etypes
    src, dst = _find_edges(local_g, partition_book, seed_edges)
    src_tids, _ = partition_book.map_to_per_ntype(src)
    dst_tids, _ = partition_book.map_to_per_ntype(dst)
    canonical_etypes = {}
    for src_tid, etype_id, dst_tid in zip(src_tids, seed_edge_tids, dst_tids):
        src_tid = src_tid.item()
        etype_id = etype_id.item()
        dst_tid = dst_tid.item()
        c_etype = (ntypes[src_tid], etypes[etype_id], ntypes[dst_tid])
        canonical_etypes[canonical_etypes_delimiter.join(c_etype)] = etype_id
    return canonical_etypes


def _find_edges(local_g, partition_book, seed_edges):
    local_eids = partition_book.eid2localeid(seed_edges, partition_book.partid)
    local_src, local_dst = local_g.find_edges(local_eids)
    global_nid_mapping = local_g.ndata[dgl.NID]
    global_src = global_nid_mapping[local_src]
    global_dst = global_nid_mapping[local_dst]
    return global_src, global_dst


def is_old_version(config):
    first_etype = list(config[etypes_key].keys())[0]
    etype_tuple = first_etype.split(canonical_etypes_delimiter)
    return len(etype_tuple) == 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Change edge type in config file from format (str)"
        " to (str,str,str), the original file will be overwritten",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--part-config", type=str, help="The file of the partition config"
    )
    args, _ = parser.parse_known_args()
    assert (
        args.part_config is not None
    ), "A user has to specify a partition config file with --part_config."

    start = time.time()
    convert_conf(args.part_config)
    end = time.time()
    logging.info(f"elplased time in seconds: {end - start}")
