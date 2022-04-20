import argparse
import numpy as np
import torch.multiprocessing as mp

from initialize import proc_exec, init_process 

def log_params(params): 
    """
    Print all the arguments for debugging purposes.

    Parameters:
    -----------
    params: Argument Parser structure listing all the pre-defined parameters
    """

    print('Input Dir: ', params.input_dir)
    print('Graph Name: ', params.graph_name)
    print('Schema File: ', params.schema)
    print('No. partitions: ', params.num_parts)
    print('No. node weights: ', params.num_node_weights)
    print('Workspace dir: ', params.workspace)
    print('Node Attr Type: ', params.node_attr_dtype)
    print('Edge Attr Dtype: ', params.edge_attr_dtype)
    print('Output Dir: ', params.output)
    print('Removed Edges File: ', params.removed_edges)
    print('WorldSize: ', params.world_size)
    print('Nodes File: ', params.nodes_file)
    print('Edges File: ', params.edges_file)
    print('Node feats: ', params.node_feats_file)
    print('Edge feats: ', params.edge_feats_file)
    print('Metis partitions: ', params.metis_partitions)


def start_local_run(params): 
    """
    Function designed to run distributed implementation on a single machine

    Parameters:
    -----------
    params : Argument Parser structure with pre-determined arguments as defined
             at the bottom of this file.
    """

    log_params(params)
    processes = []
    mp.set_start_method("spawn")

    #Invoke `target` function from each of the spawned process for distributed 
    #implementation
    for rank in range(params.world_size):
        p = mp.Process(target=init_process, args=(rank, params.world_size, proc_exec, params))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


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
    parser.add_argument('--num-node-weights', required=True, type=int,
                     help='The number of node weights used by METIS.')
    parser.add_argument('--workspace', type=str, default='/tmp',
                    help='The directory to store the intermediate results')
    parser.add_argument('--node-attr-dtype', type=str, default=None,
                    help='The data type of the node attributes')
    parser.add_argument('--edge-attr-dtype', type=str, default=None,
                    help='The data type of the edge attributes')
    parser.add_argument('--output', required=True, type=str,
                    help='The output directory of the partitioned results')
    parser.add_argument('--removed-edges', help='a file that contains the removed self-loops and duplicated edges',
                    default=None, type=str)

    #arguments needed for the distributed implementation
    parser.add_argument('--world-size', help='no. of processes to spawn',
                    default=1, type=int, required=True)
    parser.add_argument('--nodes-file', help='filename of the nodes metadata', 
                    default=None, type=str, required=True)
    parser.add_argument('--edges-file', help='filename of the nodes metadata', 
                    default=None, type=str, required=True)
    parser.add_argument('--node-feats-file', help='filename of the nodes features', 
                    default=None, type=str, required=True)
    parser.add_argument('--edge-feats-file', help='filename of the nodes metadata', 
                    default=None, type=str )
    parser.add_argument('--metis-partitions', help='filename of the output of dgl_part2 (metis partitions)',
                    default=None, type=str)
    params = parser.parse_args()

    start_local_run(params)
