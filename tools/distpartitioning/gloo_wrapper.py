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
    rank_sizes = np.zeros(world_size + 1, dtype=np.int64)
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

def alltoallv_cpu_data(rank, world_size, input_tensor_list, val_dtype):
    """
    Wrapper function to providing the alltoallv functionality by using underlying alltoall
    messaging primitive

    Parameters:
    -----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    input_tensor_list : List of tensor
        The tensors to exchange
    val_dtype : torch dtype
        torch dtype which is used to create buffer for receiving messages

    Returns:
    --------
    list : 
        list of tensors received from other processes during alltoall message

    """
    sizes = [list(x.size()) for x in input_tensor_list]
    for idx in range(1,len(sizes)):
        assert len(sizes[idx-1]) == len(sizes[idx])

    #decide how much to pad. 
    #always use the first-dimension for padding. 
    ll = [ x[0] for x in sizes ]

    #dims of the outgoing messages
    out_dims = [ [np.amax(ll)] + l[1:] for idx, l in enumerate(sizes) ]

    #dims of the padding needed, if any
    diff_dims = [ [np.amax(ll) - l[0]] + l[1:] for l in sizes ]

    #pad the actual message
    input_tensor_list = [torch.cat((x, torch.zeros(diff_dims[idx]).type(x.dtype))) for idx, x in enumerate(input_tensor_list)]

    #send useful message sizes to all
    send_counts = []
    recv_counts = []
    for idx in range(world_size):
        #send a 3 element triplet, [a, b, ....] where 
        #a = useful message dim, b = amount of padding and remaining elements are
        #the remaining dimensions of the tensor
        send_counts.append(torch.from_numpy(np.array([sizes[idx][0]] + out_dims[idx])).type(torch.int64))
        recv_counts.append(torch.zeros((1 + len(sizes[idx])), dtype=torch.int64))
    alltoall_cpu(rank, world_size, recv_counts, send_counts)

    #allocate buffers for receiving message
    output_tensor_list = []
    recv_counts = [ tsize.numpy() for tsize in recv_counts]
    for tsize in recv_counts:
        output_tensor_list.append(torch.zeros(tuple(tsize[1:])).type(val_dtype))

    #send actual message itself. 
    alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list)

    #extract un-padded message from the output_tensor_list and return it
    return_vals = []
    for s, t in zip(recv_counts, output_tensor_list):
        if s[0] == 0:
            return_vals.append(None)
        else:
            return_vals.append(t[0:s[0]])
    return return_vals

def alltoall_cpu_object_lst(rank, world_size, input_list):
    """
    Each process scatters list of input objects to all processes in a cluster
    and return gathered list of objects in output list. 

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    input_tensor_list : List of tensor
        The tensors to exchange

    Returns
    -------
    list: list of objects are received from other processes
       This is the list of objects which are sent to the current process by
       other processes as part of this exchange
    """
    rcv_list = []
    output_list = [None] * world_size
    for i in range(world_size):
        rcv_list.clear()
        rcv_list.append(None)
        if (i == rank):
            dist.scatter_object_list(rcv_list, input_list, src = rank)
        else:
            send_list = [None] * world_size
            dist.scatter_object_list(rcv_list, send_list, src = i)
        output_list[i] = rcv_list[0]

    return output_list

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
