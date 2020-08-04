"""Initialize the distributed services"""

import multiprocessing as mp
import traceback
import atexit
from . import rpc
from .constants import MAX_QUEUE_SIZE
from .kvstore import init_kvstore, close_kvstore
from .rpc_client import connect_to_server, shutdown_servers

SAMPLER_POOL = None
NUM_SAMPLER_WORKERS = 0
INITIALIZED = False


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
    """Init rpc service"""
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
    if SAMPLER_POOL is not None:
        SAMPLER_POOL.close()
        SAMPLER_POOL.join()
    global INITIALIZED
    INITIALIZED = False


def is_initialized():
    """Is RPC initialized?
    """
    return INITIALIZED


def exit_client():
    """Register exit callback.
    """
    # Only client with rank_0 will send shutdown request to servers.
    rpc.client_barrier()
    shutdown_servers()
    finalize_client()
    close_kvstore()
    atexit.unregister(exit_client)
