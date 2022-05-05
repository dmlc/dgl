import os
import sys
import math
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from initialize import proc_exec

def multi_dev_init(params):
    """
    Function to be invoked when executing data loading pipeline on multiple machines

    Parameters:
    -----------
    params : argparser object
        argparser object providing access to command line arguments.
    """
    #init the gloo process group here. 
    dist.init_prcess_group("gloo", rank=params.rank, world_size=params.world_size)
    print('[Rank: ', params.rank, '] Done with process group initialization...')

    #invoke the main function here.
    proc_exec(params.rank, params.world_size, params)
    print('[Rank: ', params.rank, '] Done with Distributed data processing pipeline processing.')
