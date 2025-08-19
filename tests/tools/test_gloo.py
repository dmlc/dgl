import os
from datetime import timedelta

from enum import Enum

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from distpartitioning.gloo_wrapper import (
    allgather_sizes,
    alltoallv_cpu,
    gather_metadata_json,
)


class TEST_TYPE(Enum):
    TEST_ALLGATHER_SIZES = 0
    TEST_ALLTOALLV = 1
    TEST_METADATA = 2


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


def _init_process_group(rank, world_size):
    """Function to init process group

    Parameters:
    ----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    """
    # init the gloo process group here.
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=180),
    )
    print(f"[Rank: {rank}] Done with process group initialization...")


def run_allgather_sizes(rank, world_size, num_parts, return_sizes):
    """Function to test the results of gloo_wrapper:allgather_sizes

    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    num_parts : int
        no of output graph partitions
    return_sizes : bool
        bool to indicate to return raw data as received when alltoallv_cpu
        is called
    """
    send_data = [rank]
    recv_data = allgather_sizes(send_data, world_size, num_parts, return_sizes)

    # validate the results
    if return_sizes:
        expected_result = np.arange(world_size)
    else:
        # get the offset by using each ranks row-count
        expected_result = [0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45, 55][
            0 : (world_size + 1)
        ]
    assert np.all(
        expected_result == recv_data
    ), f"Expected and Actual results do not match"


def run_alltoallv(rank, world_size, retain_nones):
    """Function to test gloo_wrapper::alltoallv_cpu and validate the
    results

    Parameters:
    -----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    retain_nones : bool
        bool to indicate whether to return NoneTypes when allgather_sizes
        is called
    """
    # gen. output data for all other ranks
    input_tensor_list = []
    for idx in range(world_size):
        input_tensor_list.append(
            torch.as_tensor(np.arange(rank, rank + 10, dtype=np.int64))
        )

    # fire test case
    output_tensors = alltoallv_cpu(
        rank, world_size, input_tensor_list, retain_nones=retain_nones
    )

    # validate output
    ranked_output = [t.numpy() for t in output_tensors]
    for idx in range(world_size):
        print(f"[Rank: {rank} - received data is {ranked_output[idx]}")
        received_data = ranked_output[idx]
        expected_data = np.arange(idx, idx + 10)
        assert np.all(
            received_data == expected_data
        ), f"Assert failed when comparing received and expected results"


def run_metadata(rank, world_size):
    """Function to test allgather_metadata_json function and
    validate the results

    Parameters:
    ----------
    rank : int
        id of the process
    world_size : int
        no. of processes to spawn
    """
    # gen. test data
    metadata = {
        "graph_name": "test-graph",
        "num_nodes": [10],
        "num_edges": [10, 20],
        "part_method": "metis",
        "num_parts": 2,
        "halo_hops": 1,
    }

    # fire test case
    all_metadata = gather_metadata_json(metadata, rank, world_size)
    all_metadata[rank] = metadata

    # validate output
    for idx in range(world_size):
        assert all_metadata[idx]["graph_name"] == "test-graph"
        assert all_metadata[idx]["num_nodes"] == [10]
        assert all_metadata[idx]["num_edges"] == [10, 20]
        assert all_metadata[idx]["part_method"] == "metis"
        assert all_metadata[idx]["num_parts"] == 2
        assert all_metadata[idx]["halo_hops"] == 1


def _run(
    port_num,
    rank,
    world_size,
    num_parts,
    return_sizes,
    retain_nones,
    test_type,
    sh_dict,
):
    """Main function for the child processes to run unit tests

    Parameters:
    ----------
    port_num : int
        port to use for communication
    rank : int
        id of the process
    world_size : int
        total no. of processes to use
    num_parts : int
        no. of output graph partitions
    return_sizes : bool
        bool to indicate to return raw data as received when alltoallv_cpu
        is called
    retain_nones : bool
        bool to indicate whether to return NoneTypes when allgather_sizes
        is called
    test_type : enum
        indicates which test case to run
    sh_dict : dict
        shared dictionary to return errors from child processes to the
        parent process
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port_num)
    _init_process_group(rank, world_size)

    try:
        if test_type == TEST_TYPE.TEST_ALLGATHER_SIZES:
            run_allgather_sizes(rank, world_size, num_parts, return_sizes)
        elif test_type == TEST_TYPE.TEST_ALLTOALLV:
            run_alltoallv(rank, world_size, retain_nones)
        elif test_type == TEST_TYPE.TEST_METADATA:
            run_metadata(rank, world_size)

    except Exception as arg:
        sh_dict[f"RANK-{rank}"] = inst


def _single_machine_run(
    world_size, num_parts, return_sizes=None, retain_nones=None, test_type=None
):
    """Main function for the parent process

    Parameters
    ----------
    world_size : int
        no. of processes to spawn
    num_parts : int
        no. of output graph partitions
    retain_sizes : bool
        if true, then shapes received are return as it is.
        otherwise prefix sum is returned, when alltoallv_cpu is called
    retain_nones : bool
        indicates to return NoneTypes to the caller, when allgather_sizes is
        invoked
    test_type : enum
        indicates  the type of the unit test to run
    """
    port_num = np.random.randint(10000, 20000, size=(1,), dtype=int)[0]
    ctx = mp.get_context("spawn")
    manager = mp.Manager()

    # shared dictionary to store any assertion failures in spawned processes
    sh_dict = manager.dict()
    processes = []
    for rank in range(world_size):
        p = ctx.Process(
            target=_run,
            args=(
                port_num,
                rank,
                world_size,
                num_parts,
                return_sizes,
                retain_nones,
                test_type,
                sh_dict,
            ),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        p.close()

    # Make sure that the spawned process, mimicing ranks/workers, did
    # not generate any errors or assertion failures
    assert len(sh_dict) == 0, f"Spawned processes reported some errors !!!"


@pytest.mark.parametrize(
    "world_size, num_parts",
    [
        [4, 4],
    ],
)
@pytest.mark.parametrize("return_sizes", [True, False])
def test_allgather_sizes(world_size, num_parts, return_sizes):
    """Unit test for testing gloo_wrapper::allgather_sizes

    Parameters:
    -----------
    world_size : int
        no. of processes to spawn
    num_parts : int
        no. of output graph partitions
    retain_sizes : bool
        if true, then shapes received are return as it is.
        otherwise prefix sum is returned
    """
    _single_machine_run(world_size, num_parts, return_sizes)


@pytest.mark.parametrize(
    "world_size, num_parts",
    [
        [4, 4],
    ],
)
@pytest.mark.parametrize("retain_nones", [True, False])
def test_alltoallv_cpu(world_size, num_parts, retain_nones):
    """Unit test for testing gloo_wrapper::alltoallv_cpu

    Parameters:
    ----------
    world_size : int
        no. of processes to spawn
    num_parts : int
        no. of output graph partitions
    retain_nones : bool
        indicates whether to return NoneTypes to the function caller
        if NoneType is received during the alltoallv_cpu function
    """
    _single_machine_run(
        world_size,
        num_parts,
        retain_nones=retain_nones,
        test_type=TEST_TYPE.TEST_ALLTOALLV,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_gather_metadata_json(world_size):
    """Unit test for testing  gloo_wrapper::gather_metadata_json

    Parameters:
    ----------
    world_size : int
        no. of processes to spawn
    """
    _single_machine_run(world_size, 0, test_type=TEST_TYPE.TEST_METADATA)
