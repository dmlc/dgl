import numpy as np
import torch
import torch.distributed as dist


def allgather_sizes(send_data, world_size, num_parts, return_sizes=False):
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
    num_parts : integer
        No. of output graph partitions
    return_sizes : bool
        Boolean flag to indicate whether to return raw sizes from each process
        or perform prefix sum on the raw sizes.

    Returns :
    ---------
        numpy array
            array with the prefix sum
    """

    # Assert on the world_size, num_parts
    assert (num_parts % world_size) == 0

    # compute the length of the local data
    send_length = len(send_data)
    out_tensor = torch.as_tensor(send_data, dtype=torch.int64)
    in_tensor = [
        torch.zeros(send_length, dtype=torch.int64) for _ in range(world_size)
    ]

    # all_gather message
    dist.all_gather(in_tensor, out_tensor)

    # Return on the raw sizes from each process
    if return_sizes:
        return torch.cat(in_tensor).numpy()

    # gather sizes in on array to return to the invoking function
    rank_sizes = np.zeros(num_parts + 1, dtype=np.int64)
    part_counts = torch.cat(in_tensor).numpy()

    count = rank_sizes[0]
    idx = 1
    for local_part_id in range(num_parts // world_size):
        for r in range(world_size):
            count += part_counts[r * (num_parts // world_size) + local_part_id]
            rank_sizes[idx] = count
            idx += 1

    return rank_sizes


def __alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list):
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
    input_tensor_list = [
        tensor.to(torch.device("cpu")) for tensor in input_tensor_list
    ]
    # TODO(#5002): As Boolean data is not supported in
    # ``torch.distributed.scatter()``, we convert boolean into uint8 before
    # scatter and convert it back afterwards.
    dtypes = [t.dtype for t in input_tensor_list]
    for i, dtype in enumerate(dtypes):
        if dtype == torch.bool:
            input_tensor_list[i] = input_tensor_list[i].to(torch.int8)
            output_tensor_list[i] = output_tensor_list[i].to(torch.int8)
    for i in range(world_size):
        dist.scatter(
            output_tensor_list[i], input_tensor_list if i == rank else [], src=i
        )
    # Convert back to original dtype
    for i, dtype in enumerate(dtypes):
        if dtype == torch.bool:
            input_tensor_list[i] = input_tensor_list[i].to(dtype)
            output_tensor_list[i] = output_tensor_list[i].to(dtype)


def alltoallv_cpu(rank, world_size, input_tensor_list, retain_nones=True):
    """
    Wrapper function to providing the alltoallv functionality by using underlying alltoall
    messaging primitive. This function, in its current implementation, supports exchanging
    messages of arbitrary dimensions and is not tied to the user of this function.

    This function pads all input tensors, except one, so that all the messages are of the same
    size. Once the messages are padded, It first sends a vector whose first two elements are
    1) actual message size along first dimension, and 2) Message size along first dimension
    which is used for communication. The rest of the dimensions are assumed to be same across
    all the input tensors. After receiving the message sizes, the receiving end will create buffers
    of appropriate sizes. And then slices the received messages to remove the added padding, if any,
    and returns to the caller.

    Parameters:
    -----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire
    input_tensor_list : List of tensor
        The tensors to exchange
    retain_nones : bool
        Indicates whether to retain ``None`` data in returned value.

    Returns:
    --------
    list :
        list of tensors received from other processes during alltoall message

    """
    # ensure len of input_tensor_list is same as the world_size.
    assert input_tensor_list != None
    assert len(input_tensor_list) == world_size

    # ensure that all the tensors in the input_tensor_list are of same size.
    sizes = [list(x.size()) for x in input_tensor_list]
    for idx in range(1, len(sizes)):
        assert len(sizes[idx - 1]) == len(
            sizes[idx]
        )  # no. of dimensions should be same
        assert (
            input_tensor_list[idx - 1].dtype == input_tensor_list[idx].dtype
        )  # dtype should be same
        assert (
            sizes[idx - 1][1:] == sizes[idx][1:]
        )  # except first dimension remaining dimensions should all be the same

    # decide how much to pad.
    # always use the first-dimension for padding.
    ll = [x[0] for x in sizes]

    # dims of the padding needed, if any
    # these dims are used for padding purposes.
    diff_dims = [[np.amax(ll) - l[0]] + l[1:] for l in sizes]

    # pad the actual message
    input_tensor_list = [
        torch.cat((x, torch.zeros(diff_dims[idx]).type(x.dtype)))
        for idx, x in enumerate(input_tensor_list)
    ]

    # send useful message sizes to all
    send_counts = []
    recv_counts = []
    for idx in range(world_size):
        # send a vector, of atleast 3 elements, [a, b, ....] where
        # a = useful message dim, b = actual message outgoing message size along the first dimension
        # and remaining elements are the remaining dimensions of the tensor
        send_counts.append(
            torch.from_numpy(
                np.array([sizes[idx][0]] + [np.amax(ll)] + sizes[idx][1:])
            ).type(torch.int64)
        )
        recv_counts.append(
            torch.zeros((1 + len(sizes[idx])), dtype=torch.int64)
        )
    __alltoall_cpu(rank, world_size, recv_counts, send_counts)

    # allocate buffers for receiving message
    output_tensor_list = []
    recv_counts = [tsize.numpy() for tsize in recv_counts]
    for idx, tsize in enumerate(recv_counts):
        output_tensor_list.append(
            torch.zeros(tuple(tsize[1:])).type(input_tensor_list[idx].dtype)
        )

    # send actual message itself.
    __alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list)

    # extract un-padded message from the output_tensor_list and return it
    return_vals = []
    for s, t in zip(recv_counts, output_tensor_list):
        if s[0] == 0:
            if retain_nones:
                return_vals.append(None)
        else:
            return_vals.append(t[0 : s[0]])
    return return_vals


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

    # Populate input obj and output obj list on rank-0 and non-rank-0 machines
    input_obj = None if rank == 0 else metadata
    output_objs = [None for _ in range(world_size)] if rank == 0 else None

    # invoke the gloo method to perform gather on rank-0
    dist.gather_object(input_obj, output_objs, dst=0)
    return output_objs
