import torch
import dgl
from ..utils import cuda
from collections import Counter


def random_walk_sampler(G, nodeset, restart_prob, max_nodes):
    '''
    G: DGLGraph
    nodeset: 1D CPU Tensor of node IDs
    restart_prob: float
    max_nodes: int
    return: list[list[Tensor]]
    '''
    traces = dgl.contrib.sampling.bipartite_single_sided_random_walk_with_restart(
            G, nodeset, restart_prob, max_nodes)

    return traces

# Note: this function is not friendly to giant graphs since we use a matrix
# with size (num_nodes_in_nodeset, num_nodes_in_graph).
def random_walk_distribution(G, nodeset, restart_prob, max_nodes):
    n_nodes = nodeset.shape[0]
    n_available_nodes = G.number_of_nodes()
    traces = random_walk_sampler(G, nodeset, restart_prob, max_nodes)
    visited_counts = torch.zeros(n_nodes, n_available_nodes)
    for i in range(n_nodes):
        visited_nodes = torch.cat(traces[i])
        visited_counts[i].scatter_add_(0, visited_nodes, torch.ones_like(visited_nodes, dtype=torch.float32))
    return visited_counts


def random_walk_distribution_topt(G, nodeset, restart_prob, max_nodes, top_T):
    '''
    returns the top T important neighbors of each node in nodeset, as well as
    the weights of the neighbors.
    '''
    visited_prob = random_walk_distribution(G, nodeset, restart_prob, max_nodes)
    weights, nodes = visited_prob.topk(top_T, 1)
    weights = weights / weights.sum(1, keepdim=True)
    return weights, nodes


def random_walk_nodeflow(G, nodeset, n_layers, restart_prob, max_nodes, top_T):
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
        nb_weights, nb_nodes = random_walk_distribution_topt(G, cur_nodeset, restart_prob, max_nodes, top_T)
        nodeflow.insert(0, (cur_nodeset.to(dev), nb_weights.to(dev), nb_nodes.to(dev)))
        cur_nodeset = torch.cat([nb_nodes.view(-1), cur_nodeset]).unique()

    return nodeflow
