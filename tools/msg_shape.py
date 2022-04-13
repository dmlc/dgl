import torch
import torch.distributed as dist

def sendShape( toRank, dims ): 

    shapeTensor = torch.zeros( len( shape ), dtype=torch.int32 )
    for idx in range( len( shape ) ): 
        sendTensor[ idx ] = shape[ idx ]
    dist.send( sendTensor, dst=toRank )
    return

def recvShape( fromRank, dims ): 

    recvTensor = torch.zeros( dims, dtype=torch.int32 )
    dist.recv( recvTensor, src=fromRank )
    return list( map( lambda x: int(x), recvTensor ) )

