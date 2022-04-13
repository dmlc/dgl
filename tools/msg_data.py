import numpy as np
import torch
import torch.distributed as dist

def sendData( toRank, sendArr): 
    shape = sendArr.shape ()
    sendShape( toRank, len( shape ) )

    dataTensor = torch.from_numpy( sendArr )
    dist.send( dataTensor, dst=toRank )

def recvData( fromRank, dims, dtype ): 
    shape = recvShape( fromRank, dims )
    assert( len(shape) == len(dims), "Expected dimensions and received dimensions does not match !!!")
    for idx in range( len( shape ) ): 
        assert( shape[ idx ] == dims[ idx ], "Expected component dimension and received does not match !!!")

    dataTensor = torch.zeros( shape, dtype=dtype)
    dist.recv( dataTensor, src=fromRank )
    return dataTensor.numpy ()

