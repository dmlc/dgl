import torch as th
import torch.distributed as dist

def alltoall_cpu(rank, world_size, output, scatter_list):
    scatter_list = [tensor.to(th.device('cpu')) for tensor in scatter_list]
    for i in range(world_size):
        dist.scatter(output[i], scatter_list if i == rank else [], src=i)

def alltoallv_cpu(rank, world_size, gather_list, tensor_list, data_dtype):
    ret_list = [None] * world_size
    # send tensor to each target trainer using torch.distributed.isend
    # isend is async
    senders = []
    for i in range(world_size):
        if i == rank:
            gather_list[i] = tensor_list[i].to(th.device('cpu'))
        else:
            sender = dist.isend(tensor_list[i].to(th.device('cpu')), dst=i)
            senders.append(sender)

    for i in range(world_size):
        if i != rank:
            dist.recv(gather_list[i], src=i)

    th.distributed.barrier()
