"""Initialize the distributed services"""
# pylint: disable=line-too-long

import atexit
import gc
import multiprocessing as mp
import os
import queue
import sys
import time
import traceback
from enum import Enum

from .. import utils
from ..base import dgl_warning, DGLError
from . import rpc
from .constants import MAX_QUEUE_SIZE
from .kvstore import close_kvstore, init_kvstore
from .role import init_role
from .rpc_client import connect_to_server

SAMPLER_POOL = None
NUM_SAMPLER_WORKERS = 0
INITIALIZED = False


def set_initialized(value=True):
    """Set the initialized state of rpc"""
    global INITIALIZED
    INITIALIZED = value


def get_sampler_pool():
    """Return the sampler pool and num_workers"""
    return SAMPLER_POOL, NUM_SAMPLER_WORKERS


def _init_rpc(
    ip_config,
    num_servers,
    max_queue_size,
    role,
    num_threads,
    group_id,
):
    """This init function is called in the worker processes."""
    try:
        utils.set_num_threads(num_threads)
        if os.environ.get("DGL_DIST_MODE", "standalone") != "standalone":
            connect_to_server(ip_config, num_servers, max_queue_size, group_id)
        init_role(role)
        init_kvstore(ip_config, num_servers, role)
    except Exception as e:
        print(e, flush=True)
        traceback.print_exc()
        raise e


class MpCommand(Enum):
    """Enum class for multiprocessing command"""

    INIT_RPC = 0  # Not used in the task queue
    SET_COLLATE_FN = 1
    CALL_BARRIER = 2
    DELETE_COLLATE_FN = 3
    CALL_COLLATE_FN = 4
    CALL_FN_ALL_WORKERS = 5
    FINALIZE_POOL = 6


def init_process(rpc_config, mp_contexts):
    """Work loop in the worker"""
    try:
        _init_rpc(*rpc_config)
        keep_polling = True
        data_queue, task_queue, barrier = mp_contexts
        collate_fn_dict = {}

        while keep_polling:
            try:
                # Follow https://github.com/pytorch/pytorch/blob/d57ce8cf8989c0b737e636d8d7abe16c1f08f70b/torch/utils/data/_utils/worker.py#L260
                command, args = task_queue.get(timeout=5)
            except queue.Empty:
                continue
            if command == MpCommand.SET_COLLATE_FN:
                dataloader_name, func = args
                collate_fn_dict[dataloader_name] = func
            elif command == MpCommand.CALL_BARRIER:
                barrier.wait()
            elif command == MpCommand.DELETE_COLLATE_FN:
                (dataloader_name,) = args
                del collate_fn_dict[dataloader_name]
            elif command == MpCommand.CALL_COLLATE_FN:
                dataloader_name, collate_args = args
                data_queue.put(
                    (
                        dataloader_name,
                        collate_fn_dict[dataloader_name](collate_args),
                    )
                )
            elif command == MpCommand.CALL_FN_ALL_WORKERS:
                func, func_args = args
                func(func_args)
            elif command == MpCommand.FINALIZE_POOL:
                _exit()
                keep_polling = False
            else:
                raise Exception("Unknown command")
    except Exception as e:
        traceback.print_exc()
        raise e


class CustomPool:
    """Customized worker pool"""

    def __init__(self, num_workers, rpc_config):
        """
        Customized worker pool init function
        """
        ctx = mp.get_context("spawn")
        self.num_workers = num_workers
        # As pool could be used by any number of dataloaders, queues
        # should be able to take infinite elements to avoid dead lock.
        self.queue_size = 0
        self.result_queue = ctx.Queue(self.queue_size)
        self.results = {}  # key is dataloader name, value is fetched batch.
        self.task_queues = []
        self.process_list = []
        self.current_proc_id = 0
        self.cache_result_dict = {}
        self.barrier = ctx.Barrier(num_workers)
        for _ in range(num_workers):
            task_queue = ctx.Queue(self.queue_size)
            self.task_queues.append(task_queue)
            proc = ctx.Process(
                target=init_process,
                args=(
                    rpc_config,
                    (self.result_queue, task_queue, self.barrier),
                ),
            )
            proc.daemon = True
            proc.start()
            self.process_list.append(proc)

    def set_collate_fn(self, func, dataloader_name):
        """Set collate function in subprocess"""
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.SET_COLLATE_FN, (dataloader_name, func))
            )
        self.results[dataloader_name] = []

    def submit_task(self, dataloader_name, args):
        """Submit task to workers"""
        # Round robin
        self.task_queues[self.current_proc_id].put(
            (MpCommand.CALL_COLLATE_FN, (dataloader_name, args))
        )
        self.current_proc_id = (self.current_proc_id + 1) % self.num_workers

    def submit_task_to_all_workers(self, func, args):
        """Submit task to all workers"""
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.CALL_FN_ALL_WORKERS, (func, args))
            )

    def get_result(self, dataloader_name, timeout=1800):
        """Get result from result queue"""
        if dataloader_name not in self.results:
            raise DGLError(
                f"Got result from an unknown dataloader {dataloader_name}."
            )
        while len(self.results[dataloader_name]) == 0:
            dl_name, data = self.result_queue.get(timeout=timeout)
            self.results[dl_name].append(data)
        return self.results[dataloader_name].pop(0)

    def delete_collate_fn(self, dataloader_name):
        """Delete collate function"""
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.DELETE_COLLATE_FN, (dataloader_name,))
            )
        del self.results[dataloader_name]

    def call_barrier(self):
        """Call barrier at all workers"""
        for i in range(self.num_workers):
            self.task_queues[i].put((MpCommand.CALL_BARRIER, tuple()))

    def close(self):
        """Close worker pool"""
        for i in range(self.num_workers):
            self.task_queues[i].put(
                (MpCommand.FINALIZE_POOL, tuple()), block=False
            )
            time.sleep(0.5)  # Fix for early python version

    def join(self):
        """Join the close process of worker pool"""
        for i in range(self.num_workers):
            self.process_list[i].join()


def initialize(
    ip_config,
    max_queue_size=MAX_QUEUE_SIZE,
    net_type=None,
    num_worker_threads=1,
    use_graphbolt=False,
):
    """Initialize DGL's distributed module

    This function initializes DGL's distributed module. It acts differently in server
    or client modes. In the server mode, it runs the server code and never returns.
    In the client mode, it builds connections with servers for communication and
    creates worker processes for distributed sampling.

    Parameters
    ----------
    ip_config: str
        File path of ip_config file
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).

        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str, optional
        [Deprecated] Networking type, can be 'socket' only.
    num_worker_threads: int
        The number of OMP threads in each sampler process.
    use_graphbolt: bool, optional
        Whether to use GraphBolt for distributed train.

    Note
    ----
    Users have to invoke this API before any DGL's distributed API and framework-specific
    distributed API. For example, when used with Pytorch, users have to invoke this function
    before Pytorch's `pytorch.distributed.init_process_group`.
    """
    print(
        f"Initialize the distributed services with graphbolt: {use_graphbolt}"
    )
    if net_type is not None:
        dgl_warning(
            "net_type is deprecated and will be removed in future release."
        )
    if os.environ.get("DGL_ROLE", "client") == "server":
        from .dist_graph import DistGraphServer

        assert (
            os.environ.get("DGL_SERVER_ID") is not None
        ), "Please define DGL_SERVER_ID to run DistGraph server"
        assert (
            os.environ.get("DGL_IP_CONFIG") is not None
        ), "Please define DGL_IP_CONFIG to run DistGraph server"
        assert (
            os.environ.get("DGL_NUM_SERVER") is not None
        ), "Please define DGL_NUM_SERVER to run DistGraph server"
        assert (
            os.environ.get("DGL_NUM_CLIENT") is not None
        ), "Please define DGL_NUM_CLIENT to run DistGraph server"
        assert (
            os.environ.get("DGL_CONF_PATH") is not None
        ), "Please define DGL_CONF_PATH to run DistGraph server"
        formats = os.environ.get("DGL_GRAPH_FORMAT", "csc").split(",")
        formats = [f.strip() for f in formats]
        rpc.reset()
        serv = DistGraphServer(
            int(os.environ.get("DGL_SERVER_ID")),
            os.environ.get("DGL_IP_CONFIG"),
            int(os.environ.get("DGL_NUM_SERVER")),
            int(os.environ.get("DGL_NUM_CLIENT")),
            os.environ.get("DGL_CONF_PATH"),
            graph_format=formats,
            use_graphbolt=use_graphbolt,
        )
        serv.start()
        sys.exit()
    else:
        num_workers = int(os.environ.get("DGL_NUM_SAMPLER", 0))
        num_servers = int(os.environ.get("DGL_NUM_SERVER", 1))
        group_id = int(os.environ.get("DGL_GROUP_ID", 0))
        rpc.reset()
        global SAMPLER_POOL
        global NUM_SAMPLER_WORKERS
        is_standalone = (
            os.environ.get("DGL_DIST_MODE", "standalone") == "standalone"
        )
        if num_workers > 0 and not is_standalone:
            SAMPLER_POOL = CustomPool(
                num_workers,
                (
                    ip_config,
                    num_servers,
                    max_queue_size,
                    "sampler",
                    num_worker_threads,
                    group_id,
                ),
            )
        else:
            SAMPLER_POOL = None
        NUM_SAMPLER_WORKERS = num_workers
        if not is_standalone:
            assert (
                num_servers is not None and num_servers > 0
            ), "The number of servers per machine must be specified with a positive number."
            connect_to_server(
                ip_config,
                num_servers,
                max_queue_size,
                group_id=group_id,
            )
        init_role("default")
        init_kvstore(ip_config, num_servers, "default")


def finalize_client():
    """Release resources of this client."""
    if os.environ.get("DGL_DIST_MODE", "standalone") != "standalone":
        rpc.finalize_sender()
        rpc.finalize_receiver()


def _exit():
    exit_client()
    time.sleep(1)


def finalize_worker():
    """Finalize workers
    Python's multiprocessing pool will not call atexit function when close
    """
    global SAMPLER_POOL
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.close()


def join_finalize_worker():
    """join the worker close process"""
    global SAMPLER_POOL
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.join()
    SAMPLER_POOL = None


def is_initialized():
    """Is RPC initialized?"""
    return INITIALIZED


def _shutdown_servers():
    set_initialized(False)
    # send ShutDownRequest to servers
    if rpc.get_rank() == 0:  # Only client_0 issue this command
        req = rpc.ShutDownRequest(rpc.get_rank())
        for server_id in range(rpc.get_num_server()):
            rpc.send_request(server_id, req)


def exit_client():
    """Trainer exits

    This function is called automatically when a Python process exits. Normally,
    the training script does not need to invoke this function at the end.

    In the case that the training script needs to initialize the distributed module
    multiple times (so far, this is needed in the unit tests), the training script
    needs to call `exit_client` before calling `initialize` again.
    """
    # Only client with rank_0 will send shutdown request to servers.
    print(
        "Client[{}] in group[{}] is exiting...".format(
            rpc.get_rank(), rpc.get_group_id()
        )
    )
    finalize_worker()  # finalize workers should be earilier than barrier, and non-blocking
    # collect data such as DistTensor before exit
    gc.collect()
    if os.environ.get("DGL_DIST_MODE", "standalone") != "standalone":
        rpc.client_barrier()
        _shutdown_servers()
    finalize_client()
    join_finalize_worker()
    close_kvstore()
    atexit.unregister(exit_client)
