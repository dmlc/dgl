"""Provide utils for distributed sparse optimizers
"""
import torch as th
import torch.distributed as dist


def alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    input_tensor_list = [
        tensor.to(th.device("cpu")) for tensor in input_tensor_list
    ]
    for i in range(world_size):
        dist.scatter(
            output_tensor_list[i], input_tensor_list if i == rank else [], src=i
        )


def alltoallv_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
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
            output_tensor_list[i] = input_tensor_list[i].to(th.device("cpu"))
        else:
            sender = dist.isend(
                input_tensor_list[i].to(th.device("cpu")), dst=i
            )
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i)

    th.distributed.barrier()


def alltoall(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list. The tensors should have the same shape.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    if th.distributed.get_backend() == "nccl":
        th.distributed.all_to_all(output_tensor_list, input_tensor_list)
    else:
        alltoall_cpu(
            rank,
            world_size,
            output_tensor_list,
            input_tensor_list,
        )


def alltoallv(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
    and return gathered list of tensors in output list.

    Parameters
    ----------
    rank : int
        The rank of current worker
    world_size : int
        The size of the entire communicator
    output_tensor_list : List of tensor
        The received tensors
    input_tensor_list : List of tensor
        The tensors to exchange
    """
    if th.distributed.get_backend() == "nccl":
        th.distributed.all_to_all(output_tensor_list, input_tensor_list)
    else:
        alltoallv_cpu(
            rank,
            world_size,
            output_tensor_list,
            input_tensor_list,
        )
