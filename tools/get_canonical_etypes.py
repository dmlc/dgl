import dgl
from dgl.distributed import graph_services, load_partition_book
import dgl.backend as F
import multiprocessing as mp
import time
import argparse
import os
import torch
from dgl.data.utils import load_graphs

def get_canonical_etypes(folder, graph_name, process_num):
    mp.set_start_method('spawn')
    gpb, _, _, etypes =  load_partition_book(part_config=f'{folder}/{graph_name}.json', part_id=0)
    eid = []
    etype_ids = []
    for etype in etypes:
        #type_eid = dgl.random.choice(50000, 1)
        type_eid = F.zeros((1,), F.int64, F.cpu())
        eid.append(gpb.map_to_homo_eid(type_eid, etype))
        etype_ids.append(etypes[etype])
    eid = F.cat(eid, 0)
    etype_ids = torch.IntTensor(etype_ids)
    partition_id = gpb.eid2partid(eid)
    canonical_etypes = []
    with mp.Pool(processes=process_num) as pool:
        pids = [pid for pid in range(gpb.num_partitions()) if pid in partition_id]
        seed_edges = [F.boolean_mask(eid, partition_id == pid) for pid in pids]
        seed_edge_types = [F.boolean_mask(etype_ids, partition_id == pid) for pid in pids]
        args = [(seed_edge, seed_edge_type, pid, folder, graph_name)
                        for pid, seed_edge, seed_edge_type
                        in zip(pids, seed_edges, seed_edge_types)]
        # for arg in args:
        #     canonical_etypes += _find_c_etypes_in_partition(*arg)
        for c_etype in pool.starmap(_find_c_etypes_in_partition, args):
            canonical_etypes += c_etype
    return canonical_etypes  

def _find_c_etypes_in_partition(seed_edges, seed_edge_types, part_id, folder, graph_name):
    try:
        local_g = load_graphs(f'{folder}/part{part_id}/graph.dgl')[0][0]
        partition_book = load_partition_book(part_config=f'{folder}/{graph_name}.json', part_id=part_id)[0]
        ntypes, etypes = partition_book.ntypes, partition_book.etypes
        src, dst = graph_services._find_edges(local_g, partition_book, seed_edges)
        src_tids, _ = partition_book.map_to_per_ntype(src)
        dst_tids, _ = partition_book.map_to_per_ntype(dst)
        canonical_etypes = []
        for src_tid, etype_id, dst_tid in zip(src_tids, seed_edge_types, dst_tids):
            src_tid = F.as_scalar(src_tid)
            etype_id = F.as_scalar(etype_id)
            dst_tid = F.as_scalar(dst_tid)
            canonical_etypes.append((ntypes[src_tid], etypes[etype_id],
                                            ntypes[dst_tid]))              
        return canonical_etypes
    except:
        print("Exception occured!!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Get canonical etypes from graph data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--in-dir', type=str, help='Location of the input directory where the partitioned graph is located')
    parser.add_argument('--graph-name', type=str, help='Location of the input directory where the partitioned graph is located')
    parser.add_argument('--process-num', type=int, default=2, help='Max number of running processes, each for one partition')

    args, _ = parser.parse_known_args()
    
    assert os.path.isdir(args.in_dir)
    assert isinstance(args.process_num, int)
    
    start = time.time()
    canonical_etypes = [','.join(c_etype) for c_etype in
                            get_canonical_etypes(args.in_dir, args.graph_name, args.process_num)]
    with open('canonical_etypes.txt','w') as file:
        file.write('\n'.join(canonical_etypes))
    end = time.time()
    print(f'elplased time in seconds: {end - start}')
