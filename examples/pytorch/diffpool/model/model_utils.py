import torch as th
from torch.autograd import Function


def batch2tensor(batch_adj, batch_feat, node_per_pool_graph):
    """
    transform a batched graph to batched adjacency tensor and node feature tensor
    """
    batch_size = int(batch_adj.size()[0] / node_per_pool_graph)
    adj_list = []
    feat_list = []
    for i in range(batch_size):
        start = i * node_per_pool_graph
        end = (i + 1) * node_per_pool_graph
        adj_list.append(batch_adj[start:end, start:end])
        feat_list.append(batch_feat[start:end, :])
    adj_list = list(map(lambda x: th.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x: th.unsqueeze(x, 0), feat_list))
    adj = th.cat(adj_list, dim=0)
    feat = th.cat(feat_list, dim=0)

    return feat, adj


def masked_softmax(
    matrix, mask, dim=-1, memory_efficient=True, mask_fill_value=-1e32
):
    """
    masked_softmax for dgl batch graph
    code snippet contributed by AllenNLP (https://github.com/allenai/allennlp)
    """
    if mask is None:
        result = th.nn.functional.softmax(matrix, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < matrix.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            result = th.nn.functional.softmax(matrix * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_matrix = matrix.masked_fill(
                (1 - mask).byte(), mask_fill_value
            )
            result = th.nn.functional.softmax(masked_matrix, dim=dim)
    return result
