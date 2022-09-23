import json
from dgl.distributed import graph_services, load_partition_book
import dgl.backend as F
import multiprocessing as mp
import time
import argparse
import os
import torch
from dgl.data.utils import load_graphs

def etype2canonical_etypes(part_config, process_num):
    mp.set_start_method('spawn')
    gpb, _, _, etypes =  load_partition_book(part_config=part_config, part_id=0)
    eid = []
    etype_ids = []
    for etype in etypes:
        type_eid = F.zeros((1,), F.int64, F.cpu())
        eid.append(gpb.map_to_homo_eid(type_eid, etype))
        etype_ids.append(etypes[etype])
    eid = F.cat(eid, 0)
    etype_ids = torch.IntTensor(etype_ids)
    partition_id = gpb.eid2partid(eid)
    canonical_etypes = {}
    with mp.Pool(processes=process_num) as pool:
        pids = [pid for pid in range(gpb.num_partitions()) if pid in partition_id]
        seed_edges = [F.boolean_mask(eid, partition_id == pid) for pid in pids]
        seed_edge_tids = [F.boolean_mask(etype_ids, partition_id == pid) for pid in pids]
        args = [(seed_edge, seed_edge_type, pid, part_config)
                        for pid, seed_edge, seed_edge_type
                        in zip(pids, seed_edges, seed_edge_tids)]
        for c_etype in pool.starmap(_find_c_etypes_in_partition, args):
            canonical_etypes = canonical_etypes | c_etype
    return canonical_etypes

def _find_c_etypes_in_partition(seed_edges, seed_edge_tids, part_id, part_config):
    folder, _ = os.path.split(part_config)
    local_g = load_graphs(f'{folder}/part{part_id}/graph.dgl')[0][0]
    partition_book = load_partition_book(part_config=part_config, part_id=part_id)[0]
    ntypes, etypes = partition_book.ntypes, partition_book.etypes
    src, dst = graph_services._find_edges(local_g, partition_book, seed_edges)
    src_tids, _ = partition_book.map_to_per_ntype(src)
    dst_tids, _ = partition_book.map_to_per_ntype(dst)
    canonical_etypes = {}
    for src_tid, etype_id, dst_tid in zip(src_tids, seed_edge_tids, dst_tids):
        src_tid = F.as_scalar(src_tid)
        etype_id = F.as_scalar(etype_id)
        dst_tid = F.as_scalar(dst_tid)
        canonical_etypes[etype_id] = (ntypes[src_tid], etypes[etype_id],
                                        ntypes[dst_tid])
    return canonical_etypes

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get canonical etypes from graph data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--part-config', type=str, help='The file of the partition config')
    parser.add_argument('--process-num', type=int, default=1,
                        help='Number of max running processes, each one load a different partition, \
                        actual usage is min(process_num, num_partitions) \
                        please be careful when set a large number as it may exceed memory limitations')

    args, _ = parser.parse_known_args()
    
    assert os.path.isfile(args.part_config), \
            'A user has to specify a partition configuration file with --part_config.'
    assert isinstance(args.process_num, int), \
            'Process number should be int.'
    
    start = time.time()
    etypes_key = "etypes"
    canonical_etypes_key = "canonical_etypes"
    with open(args.part_config, 'r+', encoding='utf-8') as f:
        config = json.load(f)
        if canonical_etypes_key not in config:
            if etypes_key not in config:
                raise KeyError("invalid partition config, it should contain 'etypes' or ''canonical_etypes''")
            config[canonical_etypes_key] = etype2canonical_etypes(args.part_config, args.process_num)
            print(config)
            print(type(config[canonical_etypes_key]))
            del config[etypes_key]
            f.seek(0)
            json.dump(config, f, indent=4)
            f.truncate()
        end = time.time()
        print(f'elplased time in seconds: {end - start}')
