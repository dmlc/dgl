import argparse
import numpy as np
import torch.multiprocessing as mp
from data_shuffle import single_machine_run, multi_machine_run 

def log_params(params): 
    """ Print all the command line arguments for debugging purposes.

    Parameters:
    -----------
    params: argparse object
        Argument Parser structure listing all the pre-defined parameters
    """
    print('Input Dir: ', params.input_dir)
    print('Graph Name: ', params.graph_name)
    print('Schema File: ', params.schema)
    print('No. partitions: ', params.num_parts)
    print('Output Dir: ', params.output)
    print('WorldSize: ', params.world_size)
    print('Metis partitions: ', params.partitions_file)

if __name__ == "__main__":
    """ 
    Start of execution from this point. 
    Invoke the appropriate function to begin execution
    """
    #arguments which are already needed by the existing implementation of convert_partition.py
    parser = argparse.ArgumentParser(description='Construct graph partitions')
    parser.add_argument('--input-dir', required=True, type=str,
                     help='The directory path that contains the partition results.')
    parser.add_argument('--graph-name', required=True, type=str,
                     help='The graph name')
    parser.add_argument('--schema', required=True, type=str,
                     help='The schema of the graph')
    parser.add_argument('--num-parts', required=True, type=int,
                     help='The number of partitions')
    parser.add_argument('--output', required=True, type=str,
                    help='The output directory of the partitioned results')
    parser.add_argument('--partitions-dir', help='directory of the partition-ids for each node type',
                    default=None, type=str)

    #arguments needed for the distributed implementation
    parser.add_argument('--world-size', help='no. of processes to spawn',
                    default=1, type=int, required=True)
    params = parser.parse_args()

    #invoke the pipeline function
    multi_machine_run(params)
