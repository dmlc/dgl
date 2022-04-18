import torch 
import torch.distributed as dist
import itertools
import numpy as np

def alltoall_cpu(rank, world_size, output_tensor_list, input_tensor_list):
    """Each process scatters list of input tensors to all processes in a cluster
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
    """Each process scatters list of input tensors to all processes in a cluster
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
            senders.append( sender )

    for i in range(world_size):
        if i != rank:
            dist.recv(output_tensor_list[i], src=i)

    torch.distributed.barrier()

def get_global_node_ids(rank, world_size, nodeids_ranks, partitions, node_data):

    #form the count of node_ids to be send to each rank
    #get the counts to be received from all other ranks
    sizes = [ len(x) for x in nodeids_ranks]

    total_nodes = np.sum(sizes)
    if (total_nodes == 0): 
        print('Rank: ', rank, ' -- All mappings are present locally... No need for to send any info.')
        return [], []

    send_counts = list(torch.Tensor(sizes).type(dtype=torch.int32).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int32).chunk(world_size))
    alltoall_cpu(rank, world_size, recv_counts, send_counts)

    #allocate buffers to receive node-ids
    recv_nodes = []
    for i in recv_counts: 
        recv_nodes.append(torch.zeros([i.item()], dtype=torch.int32))

    #form the outgoing message
    send_nodes = []
    for i in range(world_size): 
        send_nodes.append(torch.Tensor(nodeids_ranks[i]).type(dtype=torch.int32))

    #send-recieve messages
    alltoallv_cpu(rank, world_size, recv_nodes, send_nodes)

    recv_sizes = [ x.shape for x in recv_nodes ]

    # for each of the received node-id requests lookup and send out the global node id
    send_sizes = [ len(x.tolist()) for x in recv_nodes ]
    send_counts = list(torch.Tensor(send_sizes).type(dtype=torch.int32).chunk(world_size))
    recv_counts = list(torch.zeros([world_size], dtype=torch.int32).chunk(world_size))
    alltoall_cpu( rank, world_size, recv_counts, send_counts)

    recv_global_ids = []
    for i in recv_counts: 
        recv_global_ids.append(torch.zeros([i.item()], dtype=torch.int32)) 

    # Use node_data to lookup and extract locally assigned global id to send over.
    send_nodes = []
    for i in recv_nodes: 
        #list of node-ids to lookup
        node_ids = i.tolist()
        if (len(node_ids) != 0):
            common, ind1, ind2 = np.intersect1d(node_data[:,3], node_ids, return_indices=True)
            values = node_data[ind1,0]
            send_nodes.append(torch.Tensor(values).type(dtype=torch.int32))
        else: 
            send_nodes.append(torch.Tensor([]).type(dtype=torch.int32))

    alltoallv_cpu(rank, world_size, recv_global_ids, send_nodes)

    recv_global_ids = [ x.tolist() for x in recv_global_ids ]
    global_ids = list(itertools.chain(*recv_global_ids))
    send_nodes = list(itertools.chain(*nodeids_ranks))
    print('Rank: ', 'No. of received global Ids: ', len(global_ids))

    return global_ids, send_nodes
