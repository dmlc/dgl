import logging
import math

import torch
from scipy.stats import t


def get_stats(
    array, conf_interval=False, name=None, stdout=False, logout=False
):
    """Compute mean and standard deviation from an numerical array

    Args:
        array (array like obj): The numerical array, this array can be
            convert to :obj:`torch.Tensor`.
        conf_interval (bool, optional): If True, compute the confidence interval bound (95%)
            instead of the std value. (default: :obj:`False`)
        name (str, optional): The name of this numerical array, for log usage.
            (default: :obj:`None`)
        stdout (bool, optional): Whether to output result to the terminal.
            (default: :obj:`False`)
        logout (bool, optional): Whether to output result via logging module.
            (default: :obj:`False`)
    """
    eps = 1e-9
    array = torch.Tensor(array)
    std, mean = torch.std_mean(array)
    std = std.item()
    mean = mean.item()
    center = mean

    if conf_interval:
        n = array.size(0)
        se = std / (math.sqrt(n) + eps)
        t_value = t.ppf(0.975, df=n - 1)
        err_bound = t_value * se
    else:
        err_bound = std

    # log and print
    if name is None:
        name = "array {}".format(id(array))
    log = "{}: {:.4f}(+-{:.4f})".format(name, center, err_bound)
    if stdout:
        print(log)
    if logout:
        logging.info(log)

    return center, err_bound


def get_batch_id(num_nodes: torch.Tensor):
    """Convert the num_nodes array obtained from batch graph to batch_id array
    for each node.

    Args:
        num_nodes (torch.Tensor): The tensor whose element is the number of nodes
            in each graph in the batch graph.
    """
    batch_size = num_nodes.size(0)
    batch_ids = []
    for i in range(batch_size):
        item = torch.full(
            (num_nodes[i],), i, dtype=torch.long, device=num_nodes.device
        )
        batch_ids.append(item)
    return torch.cat(batch_ids)


def topk(
    x: torch.Tensor,
    ratio: float,
    batch_id: torch.Tensor,
    num_nodes: torch.Tensor,
):
    """The top-k pooling method. Given a graph batch, this method will pool out some
    nodes from input node feature tensor for each graph according to the given ratio.

    Args:
        x (torch.Tensor): The input node feature batch-tensor to be pooled.
        ratio (float): the pool ratio. For example if :obj:`ratio=0.5` then half of the input
            tensor will be pooled out.
        batch_id (torch.Tensor): The batch_id of each element in the input tensor.
        num_nodes (torch.Tensor): The number of nodes of each graph in batch.

    Returns:
        perm (torch.Tensor): The index in batch to be kept.
        k (torch.Tensor): The remaining number of nodes for each graph.
    """
    batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

    cum_num_nodes = torch.cat(
        [num_nodes.new_zeros(1), num_nodes.cumsum(dim=0)[:-1]], dim=0
    )

    index = torch.arange(batch_id.size(0), dtype=torch.long, device=x.device)
    index = (index - cum_num_nodes[batch_id]) + (batch_id * max_num_nodes)

    dense_x = x.new_full(
        (batch_size * max_num_nodes,), torch.finfo(x.dtype).min
    )
    dense_x[index] = x
    dense_x = dense_x.view(batch_size, max_num_nodes)

    _, perm = dense_x.sort(dim=-1, descending=True)
    perm = perm + cum_num_nodes.view(-1, 1)
    perm = perm.view(-1)

    k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)
    mask = [
        torch.arange(k[i], dtype=torch.long, device=x.device)
        + i * max_num_nodes
        for i in range(batch_size)
    ]

    mask = torch.cat(mask, dim=0)
    perm = perm[mask]

    return perm, k
