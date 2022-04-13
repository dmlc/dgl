import numpy as np
import torch
import torch.distributed as dist

def executeAllGather( sendData, worldSize ): 

    sendLength = len( sendData )
    outTensor = torch.as_tensor( sendData, dtype=torch.int32 )

    inTensor = [ torch.zeros( sendLength, dtype=torch.int32 ) for _ in range(worldSize) ]

    dist.all_gather( inTensor, outTensor )

    sizesArr = []
    for t in inTensor: 
        sizesArr.append( t.item () )

    return sizesArr
