"""Fixed subgraph sampler."""
from ..sampling.utils import EidExcluder
from .base import set_node_lazy_features, set_edge_lazy_features, Sampler

# import non-DGL libraries
import numpy as np
import torch
from collections import defaultdict

class FixedSampler(Sampler):
    """Subgraph sampler that heterogeneous sampler that sets an upper 
    bound on the number of nodes included in each layer of the sampled subgraph.
    
    At each layer, the frontier is randomly subsampled. Rare node types can also be 
    upsampled by taking the scaled square root of the sampling probabilities.

    It performs node-wise neighbor sampling and returns the subgraph induced by
    all the sampled nodes.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    fixed_k : int
            The number of nodes to sample for each GNN layer.
    upsample_rare_types : bool
        Whether or not to upsample rare node types.
    replace : bool, default True
        Whether to sample with replacement
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``. The feature must be
        a scalar on each edge.
    """
    def __init__(self, fanouts, fixed_k, upsample_rare_types, replace=False, prob=None, 
                 prefetch_node_feats=None, prefetch_edge_feats=None, output_device=None):        
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.fixed_k = fixed_k
        self.upsample_rare_types = upsample_rare_types
        self.prob = prob
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(self, g, seed_nodes, exclude_eids=None):
        """Sampling function.

        Parameters
        ----------
        g : DGLGraph
            The graph to sampler from.
        seed_nodes : Tensor or dict[str, Tensor]
            The nodes sampled in the current minibatch.
        exclude_eids : Tensor or dict[etype, Tensor], optional
            The edges to exclude from neighborhood expansion.

        Returns
        -------
        input_nodes, output_nodes, subg
            A triplet containing (1) the node IDs inducing the subgraph, (2) the node
            IDs that are sampled in this minibatch, and (3) the subgraph itself.
        """

        # define empty dictionary to store reached nodes
        output_nodes = seed_nodes
        all_reached_nodes = [seed_nodes]

        # iterate over fanout
        for fanout in reversed(self.fanouts):

            # sample frontier
            frontier = g.sample_neighbors(
                seed_nodes, fanout, output_device=self.output_device,
                replace=self.replace, prob=self.prob, exclude_edges=exclude_eids)

            # get reached nodes
            curr_reached = defaultdict(list)
            for c_etype in frontier.canonical_etypes:
                (src_type, rel_type, dst_type) = c_etype
                src, _ = frontier.edges(etype = c_etype)
                curr_reached[src_type].append(src)

            # de-duplication
            curr_reached = {ntype : torch.unique(torch.cat(srcs)) for ntype, srcs in curr_reached.items()}

            # generate type sampling probabilties
            type_count = {node_type: indices.shape[0] for node_type, indices in curr_reached.items()}
            total_count = sum(type_count.values())
            probs = {node_type: count / total_count for node_type, count in type_count.items()}

            # upsample rare node types
            if self.upsample_rare_types:

                # take scaled square root of probabilities
                prob_dist = list(probs.values())
                prob_dist = np.sqrt(prob_dist)
                prob_dist = prob_dist / prob_dist.sum()

                # update probabilities
                probs = {node_type: prob_dist[i] for i, node_type in enumerate(probs.keys())}

            # generate node counts per type
            n_per_type = {node_type: int(self.fixed_k * prob) for node_type, prob in probs.items()}
            remainder = self.fixed_k - sum(n_per_type.values())
            for _ in range(remainder):
                node_type = np.random.choice(list(probs.keys()), p=list(probs.values()))
                n_per_type[node_type] += 1

            # downsample nodes
            curr_reached_k = {}
            for node_type, node_IDs in curr_reached.items():

                # get number of total nodes and number to sample
                num_nodes = node_IDs.shape[0]
                n_to_sample = min(num_nodes, n_per_type[node_type])

                # downsample nodes of current type
                random_indices = torch.randperm(num_nodes)[:n_to_sample]
                curr_reached_k[node_type] = node_IDs[random_indices]

            # update seed nodes
            seed_nodes = curr_reached_k
            all_reached_nodes.append(curr_reached_k)

        # merge all reached nodes before sending to DGLGraph.subgraph
        merged_nodes = {}
        for ntype in g.ntypes:
            merged_nodes[ntype] = torch.unique(torch.cat([reached.get(ntype, []) for reached in all_reached_nodes]))
        subg = g.subgraph(merged_nodes, relabel_nodes=True, output_device=self.output_device)

        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, subg