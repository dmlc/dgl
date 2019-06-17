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
        start = i*node_per_pool_graph
        end = (i+1)*node_per_pool_graph
        adj_list.append(batch_adj[start:end,start:end])
        feat_list.append(batch_feat[start:end,:])
    adj_list = list(map(lambda x : th.unsqueeze(x, 0), adj_list))
    feat_list = list(map(lambda x : th.unsqueeze(x, 0), feat_list))
    adj = th.cat(adj_list,dim=0)
    feat = th.cat(feat_list, dim=0)

    return feat, adj



