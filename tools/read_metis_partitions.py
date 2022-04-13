
import sys
import math
import torch
import numpy as np

def print_metis_partitions_stats( metis_map ): 
    missing_keys = []
    for i in range(len(metis_map)): 
        if i not in metis_map: 
            missing_keys.append( i )

    print( missing_keys )

    node_ids = list(metis_map.keys ()) 
    print( f'First node_id: {node_ids[0]}, Last node_id: {node_ids[-1]}')

    procIdToGlobalIdx = {}
    for i in metis_map.items (): 
        procIdToGlobalIdx.setdefault(i[1], []).append(i[0])

    for k,v in procIdToGlobalIdx.items (): 
        procIdToGlobalIdx[ k ] = v.sort()

    print(' proc-ids: ', list(procIdToGlobalIdx.keys()) )


def read_metis_partitions( metis_mapfile ): 
    metis_map = np.loadtxt( metis_mapfile, delimiter=' ', dtype='int')
    #print_metis_partitions_stats( dict(metis_map) )
    return dict( metis_map )

