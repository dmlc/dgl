"""API wrapping NCCL primitives."""

import torch
import torch.distributed as dist


def sparse_all_to_all_push(idx, value, partition):
    """Perform an all-to-all-v operation, where by all processors send out
    a set of indices and corresponding values. Indices and values,
    corresponding to the current process, will copied into the output
    arrays.

    Note: This method requires 'torch.distributed.get_backend() == "nccl"'.

    Parameters
    ----------
    idx : torch.Tensor
        The 1D set of indices to send to other processors.
    value : torch.Tensor
        The multi-dimension set of values to send to other processors.
        The first dimension must match that of `idx`.
    partition : NDArrayPartition
        The object containing information for assigning indices to
        processors.

    Returns
    -------
    torch.Tensor
        The 1D tensor of the recieved indices.
    torch.Tensor
        The set of recieved values.

    Examples
    --------

    To perform a sparse_all_to_all_push(), a partition object must be
    provided. A partition of a homgeonous graph, where the vertices are
    striped across processes can be generated via:

    >>> from dgl.partition import NDArrayPartition
    >>> part = NDArrayPartition(g.num_nodes(), world_size, mode='remainder')

    With this partition, each processor can send values to be associatd
    with vertices in the graph. So if we have an array `global_idxs` of all of
    the neighbors updated during mini-batch processing, and an array
    `global_values` containing the new values associated with the neighbors,
    we communicate them to the own processes via:

    >>> my_idxs, my_values = nccl.sparse_all_to_all_push(global_idxs, global_values, part)

    This communication pattern is common when communicating gradient
    updates for node embeddings.

    Indices the current process owns, do not need to treated specially,
    as internally they will be copied to the output array. If we have a
    set of indices in process 0 '[0, 3, 8, 9, 10]` and for process 1
    '[0, 2, 4, 5, 8, 8, 9]'. Using a remainder partition will result
    indices for processe 0 of '[0, 8, 10, 0, 2, 4, 8, 8]', and for
    process 1 of '[3, 9, 5, 9]'.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return idx, value
    assert (
        dist.get_backend() == "nccl"
    ), "requires NCCL backend to communicate CUDA tensors."

    perm, send_splits = partition.generate_permutation(idx)
    perm = perm.long()

    # Get receive splits.
    recv_splits = torch.empty_like(send_splits)
    dist.all_to_all_single(recv_splits, send_splits)

    # Use pinned memory to speedup D2H copy.
    recv_splits = recv_splits.to("cpu", non_blocking=True)
    send_splits = send_splits.to("cpu", non_blocking=True)
    send_idx = idx[perm]
    send_value = value[perm]
    # Wait D2H copy finish.
    torch.cuda.current_stream().synchronize()
    recv_sum = recv_splits.sum()
    recv_splits = recv_splits.tolist()
    send_splits = send_splits.tolist()

    # Send idx.
    recv_idx = torch.empty((recv_sum,), dtype=idx.dtype, device=idx.device)
    dist.all_to_all_single(recv_idx, send_idx, recv_splits, send_splits)

    # Send value.
    recv_value = torch.empty(
        (recv_sum, *value.shape[1:]), dtype=value.dtype, device=value.device
    )
    dist.all_to_all_single(recv_value, send_value, recv_splits, send_splits)

    return recv_idx, recv_value


def sparse_all_to_all_pull(req_idx, value, partition):
    """Perform an all-to-all-v operation, where by all processors request
    the values corresponding to their set of indices.

    Note: This method requires 'torch.distributed.get_backend() == "nccl"'.

    Parameters
    ----------
    req_idx : torch.Tensor
        The set of indices this processor is requesting.
    value : torch.Tensor
        The multi-dimension set of values that can be requested from
        this processor.
    partition : NDArrayPartition
        The object containing information for assigning indices to
        processors.

    Returns
    -------
    torch.Tensor
        The set of recieved values, corresponding to `req_idx`.

    Examples
    --------

    To perform a sparse_all_to_all_pull(), a partition object must be
    provided. A partition of a homgeonous graph, where the vertices are
    striped across processes can be generated via:

    >>> from dgl.partition import NDArrayPartition
    >>> part = NDArrayPartition(g.num_nodes(), world_size, mode='remainder')

    With this partition, each processor can request values/features
    associated with vertices in the graph. So in the case where we have
    a set of neighbors 'nbr_idxs' we need features for, and each process
    has a tensor 'node_feat' storing the features of nodes it owns in
    the partition, the features can be requested via:

    >>> nbr_values = nccl.sparse_all_to_all_pull(nbr_idxs, node_feat, part)

    Then two the arrays 'nbr_idxs' and 'nbr_values' forms the sparse
    set of features, where 'nbr_idxs[i]' is the global node id, and
    'nbr_values[i]' is the feature vector for that node. This
    communication pattern is useful for node features or node
    embeddings.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value[req_idx.long()]
    assert (
        dist.get_backend() == "nccl"
    ), "requires NCCL backend to communicate CUDA tensors."

    perm, req_splits = partition.generate_permutation(req_idx)
    perm = perm.long()

    # Get response splits.
    resp_splits = torch.empty_like(req_splits)
    dist.all_to_all_single(resp_splits, req_splits)

    # Use pinned memory to speedup D2H copy.
    resp_splits = resp_splits.to("cpu", non_blocking=True)
    req_splits = req_splits.to("cpu", non_blocking=True)
    req_idx = req_idx[perm]
    # Wait D2H copy finish.
    torch.cuda.current_stream().synchronize()
    resp_sum = resp_splits.sum()
    resp_splits = resp_splits.tolist()
    req_splits = req_splits.tolist()

    # Gather requested indices.
    resp_idx = torch.empty(
        (resp_sum,), dtype=req_idx.dtype, device=req_idx.device
    )
    dist.all_to_all_single(resp_idx, req_idx, resp_splits, req_splits)

    # Convert requested indices to local indices depending on partition.
    if resp_sum > 0:
        resp_idx = partition.map_to_local(resp_idx)

    # Collect the request value.
    req_value = torch.empty(
        (req_idx.size(0), *value.shape[1:]),
        dtype=value.dtype,
        device=value.device,
    )
    dist.all_to_all_single(req_value, value[resp_idx], req_splits, resp_splits)

    # Permute the value back into the requested order.
    return_value = torch.empty_like(req_value)
    return_value[perm] = req_value

    return return_value
