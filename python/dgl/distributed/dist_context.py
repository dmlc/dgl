"""Initialize the distributed services"""

import multiprocessing as mp
import traceback
import atexit
import time
from . import rpc
from .constants import MAX_QUEUE_SIZE
from .kvstore import init_kvstore, close_kvstore
from .rpc_client import connect_to_server, shutdown_servers

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


def _init_rpc(ip_config, max_queue_size, net_type, role):
    ''' This init function is called in the worker processes.
    '''
    try:
        connect_to_server(ip_config, max_queue_size, net_type)
        init_kvstore(ip_config, role)
    except Exception as e:
        print(e, flush=True)
        traceback.print_exc()
        raise e


def initialize(ip_config, num_workers=0, max_queue_size=MAX_QUEUE_SIZE, net_type='socket'):
    """Init rpc service
    ip_config: str
        File path of ip_config file
    num_workers: int
        Number of worker process to be created
    max_queue_size : int
        Maximal size (bytes) of client queue buffer (~20 GB on default).
        Note that the 20 GB is just an upper-bound and DGL uses zero-copy and
        it will not allocate 20GB memory at once.
    net_type : str
        Networking type. Current options are: 'socket'.
    """
    rpc.reset()
    ctx = mp.get_context("spawn")
    global SAMPLER_POOL
    global NUM_SAMPLER_WORKERS
    if num_workers > 0:
        SAMPLER_POOL = ctx.Pool(
            num_workers, initializer=_init_rpc, initargs=(ip_config, max_queue_size,
                                                          net_type, 'sampler'))
    NUM_SAMPLER_WORKERS = num_workers
    connect_to_server(ip_config, max_queue_size, net_type)
    init_kvstore(ip_config)


def finalize_client():
    """Release resources of this client."""
    rpc.finalize_sender()
    rpc.finalize_receiver()
    global INITIALIZED
    INITIALIZED = False


def _exit():
    exit_client()
    time.sleep(1)


def finalize_worker():
    """Finalize workers
       Python's multiprocessing pool will not call atexit function when close
    """
    if SAMPLER_POOL is not None:
        for _ in range(NUM_SAMPLER_WORKERS):
            SAMPLER_POOL.apply_async(_exit)
            time.sleep(0.1) # This is necessary but I don't know why
        SAMPLER_POOL.close()

def join_finalize_worker():
    """join the worker close process"""
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.join()

def is_initialized():
    """Is RPC initialized?
    """
    return INITIALIZED


def exit_client():
    """Register exit callback.
    """
    # Only client with rank_0 will send shutdown request to servers.
    finalize_worker() # finalize workers should be earilier than barrier, and non-blocking
    rpc.client_barrier()
    shutdown_servers()
    finalize_client()
    join_finalize_worker()
    close_kvstore()
    atexit.unregister(exit_client)
