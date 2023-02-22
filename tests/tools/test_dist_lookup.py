import logging
import os
import platform
import tempfile
from datetime import timedelta

import dgl

import numpy as np
import pyarrow
import pytest
import torch

import torch.distributed as dist
import torch.multiprocessing as mp

from tools.distpartitioning import constants, dist_lookup
from tools.distpartitioning.gloo_wrapper import allgather_sizes
from tools.distpartitioning.utils import get_idranges, read_json
from utils import create_chunked_dataset

try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _init_process_group(rank, world_size):
    # init the gloo process group here.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    print(f"[Rank: {rank}] Done with process group initialization...")


def _create_lookup_service(partitions_dir, ntypes, id_map, rank, world_size):
    id_lookup = dist_lookup.DistLookupService(
        partitions_dir,
        ntypes,
        id_map,
        rank,
        world_size,
    )

    # invoke the main function here.
    print(f"[Rank: {rank}] Done with Dist Lookup Service initialization...")

    return id_lookup


def _run(
    rank, num_parts, world_size, partitions_dir, ntypes, id_map, test_data
):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"

    _init_process_group(rank, world_size)
    lookup = _create_lookup_service(
        partitions_dir, ntypes, id_map, rank, world_size
    )

    tests_exec = 0
    for worker, data in test_data.items():
        if f"rank-{rank}" == worker:
            for item in data:
                method = item[0]
                request = item[1]
                response = item[2]

                if method == "getpartitionids":
                    ret_val = lookup.get_partition_ids(request)
                    tests_exec += 1
                    assert np.all(ret_val == response)
                else:
                    assert False

    # ensure all the tests are executed.
    rank_counts = allgather_sizes([tests_exec], world_size, num_parts, True)
    assert np.sum(rank_counts) == len(test_data)


def _single_machine_run(
    num_parts, world_size, partitions_dir, ntypes, id_map, test_data
):
    processes = []
    for rank in range(world_size):
        p = mp.Process(
            target=_run,
            args=(
                rank,
                num_parts,
                world_size,
                partitions_dir,
                ntypes,
                id_map,
                test_data,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def _prepare_test_data(partitions_dir, ntypes, gid_ranges, world_size):
    # read node-id to partition-id mappings from disk
    ntype_partids = []
    for ntype_id, ntype in enumerate(ntypes):
        filename = f"{ntype}.txt"
        assert os.path.isfile(os.path.join(partitions_dir, filename))

        read_options = pyarrow.csv.ReadOptions(
            use_threads=True,
            block_size=4096,
            autogenerate_column_names=True,
        )
        parse_options = pyarrow.csv.ParseOptions(delimiter=" ")

        with pyarrow.csv.open_csv(
            os.path.join(partitions_dir, "{}.txt".format(ntype)),
            read_options=read_options,
            parse_options=parse_options,
        ) as reader:
            for next_chunk in reader:
                if next_chunk is None:
                    break
                next_table = pyarrow.Table.from_batches([next_chunk])
                ntype_partids.append(next_table["f0"].to_numpy())

    # prepare test data for each rank here
    # key = f'rank-{rank}'
    # value is a list of tuple [(method-name, request, response)]
    test_data = {}
    for rank in range(world_size):
        ntype_id = np.random.randint(0, len(ntypes) - 1)
        ntype = ntypes[ntype_id]
        request = (
            np.arange(len(ntype_partids[ntype_id]))
            + gid_ranges[ntypes[ntype_id]][0, 0]
        )
        response = ntype_partids[ntype_id]
        test_data[f"rank-{rank}"] = [("getpartitionids", request, response)]

    # randomly shuffle the global-nids and retrieve their partition-ids.
    for rank in range(world_size):
        ntype_id = np.random.randint(0, len(ntypes) - 1)
        ntype = ntypes[ntype_id]
        idx = np.arange(len(ntype_partids[ntype_id]))
        np.random.shuffle(idx)
        request = idx + gid_ranges[ntypes[ntype_id]][0, 0]
        response = ntype_partids[ntype_id][idx]
        test_data[f"rank-{rank}"] = [("getpartitionids", request, response)]

    # one final test
    # mix all the ntypes and shuffle randomly
    request = []
    response = []
    for idx in range(len(ntype_partids)):
        request.append(
            np.arange(len(ntype_partids[ntype_id]))
            + gid_ranges[ntypes[ntype_id]][0, 0]
        )
        response.append(ntype_partids[idx])

    request = np.concatenate(request)
    response = np.concatenate(response)

    idx = np.arange(len(request))
    np.random.shuffle(idx)
    request = request[idx]
    response = response[idx]
    for idx in range(world_size):
        test_data[f"rank-{idx}"] = [("getpartitionids", request, response)]

    return test_data


@pytest.mark.parametrize(
    "num_chunks, num_parts, world_size",
    [[4, 4, 4], [8, 4, 2], [8, 4, 4], [9, 6, 3], [11, 11, 1], [11, 4, 1]],
)
def test_lookup_service(
    num_chunks,
    num_parts,
    world_size,
    num_chunks_nodes=None,
    num_chunks_edges=None,
    num_chunks_node_data=None,
    num_chunks_edge_data=None,
):

    with tempfile.TemporaryDirectory() as root_dir:
        g = create_chunked_dataset(
            root_dir,
            num_chunks,
            data_fmt="numpy",
            num_chunks_nodes=num_chunks_nodes,
            num_chunks_edges=num_chunks_edges,
            num_chunks_node_data=num_chunks_node_data,
            num_chunks_edge_data=num_chunks_edge_data,
        )

        # Step1: graph partition
        in_dir = os.path.join(root_dir, "chunked-data")
        output_dir = os.path.join(root_dir, "parted_data")
        os.system(
            "python3 tools/partition_algo/random_partition.py "
            "--in_dir {} --out_dir {} --num_partitions {}".format(
                in_dir, output_dir, num_parts
            )
        )

        # metadata for original graph
        orig_config = os.path.join(in_dir, "metadata.json")
        orig_schema = read_json(orig_config)
        ntypes = orig_schema[constants.STR_NODE_TYPE]

        type_nid_dict, global_nid_ranges = get_idranges(
            orig_schema[constants.STR_NODE_TYPE],
            orig_schema[constants.STR_NUM_NODES_PER_CHUNK],
            num_chunks=num_parts,
        )

        id_map = dgl.distributed.id_map.IdMap(global_nid_ranges)

        # run the test
        _single_machine_run(
            num_parts,
            world_size,
            output_dir,
            ntypes,
            id_map,
            _prepare_test_data(
                output_dir, ntypes, global_nid_ranges, world_size
            ),
        )
