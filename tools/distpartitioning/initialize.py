import os
import json
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import constants
import random

from multi_dev_init import gen_dist_partitions

def single_dev_proc_init(params):
    """ Main function for distributed implementation on a single machine

    Parameters:
    -----------
    params : argparser object
        Argument Parser structure with pre-determined arguments as defined
        at the bottom of this file.
    """
    log_params(params)
    processes = []
    mp.set_start_method("spawn")

    #Invoke `target` function from each of the spawned process for distributed
    #implementation
    for rank in range(params.world_size):
        p = mp.Process(target=run, args=(rank, params.world_size, gen_dist_partitions, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

def run(rank, world_size, func_exec, params, backend="gloo"):
    """
    Init. function which is run by each process in the Gloo ProcessGroup

    Parameters:
    -----------
    rank : integer
        rank of the process
    world_size : integer
        number of processes configured in the Process Group
    proc_exec : function name
        function which will be invoked which has the logic for each process in the group
    params : argparser object
        argument parser object to access the command line arguments
    backend : string
        string specifying the type of backend to use for communication
    """
    os.environ["MASTER_ADDR"] = '127.0.0.1'
    os.environ["MASTER_PORT"] = '29500'

    #create Gloo Process Group
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timedelta(seconds=5*60))

    #Invoke the main function to kick-off each process
    func_exec(rank, world_size, params)

def multi_dev_proc_init(params):
    """
    Function to be invoked when executing data loading pipeline on multiple machines

    Parameters:
    -----------
    params : argparser object
        argparser object providing access to command line arguments.
    """
    rank = int(os.environ["RANK"])
    
    #init the gloo process group here. 
    dist.init_process_group("gloo", rank=rank, world_size=params.world_size, timeout=timedelta(seconds=5*60))
    print('[Rank: ', rank, '] Done with process group initialization...')

    #invoke the main function here.
    gen_dist_partitions(rank, params.world_size, params)
    print('[Rank: ', rank, '] Done with Distributed data processing pipeline processing.')
