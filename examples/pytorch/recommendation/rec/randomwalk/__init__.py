import torch
import dgl
from ..utils import cuda

def random_walk_sampler(G, nodeset, n_traces, n_hops):
    '''
    G: DGLGraph
    nodeset: 1D CPU Tensor of node IDs
    n_traces: int
    n_hops: int
    return: 3D CPU Tensor or node IDs (n_nodes, n_traces, n_hops + 1)
    '''
    n_nodes = nodeset.shape[0]
    traces = torch.zeros(n_nodes, n_traces, n_hops + 1, dtype=torch.int64)

    for i in range(n_nodes):
        for j in range(n_traces):
            cur = nodeset[i]
            for k in range(n_hops + 1):
                traces[i, j, k] = cur
                neighbors = G.successors(cur.item())
                assert neighbors.shape[0] > 0
                cur = neighbors[torch.randint(len(neighbors), ())]

    return traces

# Note: this function is not friendly to giant graphs since we use a matrix
# with size (num_nodes_in_nodeset, num_nodes_in_graph).
def random_walk_distribution(G, nodeset, n_traces, n_hops):
    n_nodes = nodeset.shape[0]
    n_available_nodes = G.number_of_nodes()
    traces = random_walk_sampler(G, nodeset, n_traces, n_hops)
    visited_nodes = traces[:, :, 1:].contiguous().view(n_nodes, -1)  # (n_nodes, n_visited_other_nodes)
    visited_counts = (
            torch.zeros(n_nodes, n_available_nodes)
            .scatter_add_(1, visited_nodes, torch.ones_like(visited_nodes, dtype=torch.float32)))
    visited_prob = visited_counts / visited_counts.sum(1, keepdim=True)
    return visited_prob

def random_walk_distribution_topt(G, nodeset, n_traces, n_hops, top_T):
    '''
    returns the top T important neighbors of each node in nodeset, as well as
    the weights of the neighbors.
    '''
    visited_prob = random_walk_distribution(G, nodeset, n_traces, n_hops)
    return visited_prob.topk(top_T, 1)

def random_walk_nodeflow(G, nodeset, n_layers, n_traces, n_hops, top_T):
    '''
    returns a list of triplets (
        "active" node IDs whose embeddings are computed at the i-th layer (num_nodes,)
        weight of each neighboring node of each "active" node on the i-th layer (num_nodes, top_T)
        neighboring node IDs for each "active" node on the i-th layer (num_nodes, top_T)
    )
    '''
    dev = nodeset.device
    nodeset = nodeset.cpu()
    nodeflow = []
    cur_nodeset = nodeset
    for i in reversed(range(n_layers)):
        nb_weights, nb_nodes = random_walk_distribution_topt(G, cur_nodeset, n_traces, n_hops, top_T)
        nodeflow.insert(0, (cur_nodeset.to(dev), nb_weights.to(dev), nb_nodes.to(dev)))
        cur_nodeset = torch.cat([nb_nodes.view(-1), cur_nodeset]).unique()

    return nodeflow
