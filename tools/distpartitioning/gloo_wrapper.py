import numpy as np
import torch
import torch.distributed as dist

def allgather_sizes(send_data, world_size):
    """ 
    Perform all gather on list lengths, used to compute prefix sums
    to determine the offsets on each ranks. This is used to allocate
    global ids for edges/nodes on each ranks.

    Parameters
    ----------
    send_data : numpy array 
        Data on which allgather is performed.
    world_size : integer
        No. of processes configured for execution

    Returns : 
    ---------
        numpy array
            array with the prefix sum
    """

    #compute the length of the local data
    send_length = len(send_data)
    out_tensor = torch.as_tensor(send_data, dtype=torch.int64)
    in_tensor = [torch.zeros(send_length, dtype=torch.int64) 
                    for _ in range(world_size)]

    #all_gather message
    dist.all_gather(in_tensor, out_tensor)

    #gather sizes in on array to return to the invoking function
    rank_sizes = np.zeros(world_size + 1)
    count = rank_sizes[0]
    for i, t in enumerate(in_tensor): 
        count += t.item()
        rank_sizes[i+1] = count

    return rank_sizes

def alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """
    Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    input_tensor_list = [tensor.to(torch.device('cpu')) for tensor in input_tensor_list]
    for i in range(world_size):
        dist.scatter(output_tensor_list[i], input_tensor_list if i == rank else [], src=i)

def alltoallv_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """
    Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i in range(world_size):
        if i == rank:
            output_tensor_list[i] = input_tensor_list[i].to(torch.device('cpu'))
        else:
            sender = dist.isend(input_tensor_list[i].to(torch.device('cpu')), dst=i)
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i)

    torch.distributed.barrier()

def gather_metadata_json(metadata, rank, world_size): 
    """ 
    Gather an object (json schema on `rank`)
    Parameters:
    -----------
    metadata : json dictionary object
        json schema formed on each rank with graph level data. 
        This will be used as input to the distributed training in the later steps.
    Returns:
    --------
    list : list of json dictionary objects
        The result of the gather operation, which is the list of json dicitonary 
        objects from each rank in the world
    """

    #Populate input obj and output obj list on rank-0 and non-rank-0 machines
    input_obj = None if rank == 0 else metadata
    output_objs = [None for _ in range(world_size)] if rank == 0 else None

    #invoke the gloo method to perform gather on rank-0
    dist.gather_object(input_obj, output_objs, dst=0)
    return output_objs
