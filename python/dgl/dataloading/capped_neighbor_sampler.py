"""Capped neighbor sampler."""
from collections import defaultdict

import numpy as np
import torch

from ..sampling.utils import EidExcluder
from .base import Sampler, set_edge_lazy_features, set_node_lazy_features


class CappedNeighborSampler(Sampler):
    """Subgraph sampler that sets an upper bound on the number of nodes included in
    each layer of the sampled subgraph. At each layer, the frontier is randomly
    subsampled. Rare node types can also be upsampled by taking the scaled square
    root of the sampling probabilities. The sampler returns the subgraph induced by
    all the sampled nodes.

    This code was contributed by a community member
    ([@ayushnoori](https://github.com/ayushnoori)). There aren't currently any unit
    tests in place to verify its functionality, so please be cautious if you need
    to make any changes to the code's logic.

    Parameters
    ----------
    fanouts : list[int] or dict[etype, int]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.
        - If only a single integer is provided, DGL assumes that every edge type
            will have the same fanout.
        - If -1 is provided for one edge type on one layer, then all inbound edges
            of that edge type will be included.
    fixed_k : int
            The number of nodes to sample for each GNN layer.
    upsample_rare_types : bool
        Whether or not to upsample rare node types.
    replace : bool, default True
        Whether to sample with replacement.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``. The feature must be
        a scalar on each edge.
    """

    def __init__(
        self,
        fanouts,
        fixed_k,
        upsample_rare_types,
        replace=False,
        prob=None,
        prefetch_node_feats=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.fixed_k = fixed_k
        self.upsample_rare_types = upsample_rare_types
        self.prob = prob
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(
        self, g, indices, exclude_eids=None
    ):  # pylint: disable=arguments-differ
        """Sampling function.

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor or dict[str, Tensor]
            Nodes which induce the subgraph.
        exclude_eids : Tensor or dict[etype, Tensor], optional
            The edges to exclude from the sampled subgraph.

        Returns
        -------
        input_nodes : Tensor or dict[str, Tensor]
            The node IDs inducing the subgraph.
        output_nodes : Tensor or dict[str, Tensor]
            The node IDs that are sampled in this minibatch.
        subg : DGLGraph
            The subgraph itself.
        """

        # Define empty dictionary to store reached nodes.
        output_nodes = indices
        all_reached_nodes = [indices]

        # Iterate over fanout.
        for fanout in reversed(self.fanouts):

            # Sample frontier.
            frontier = g.sample_neighbors(
                indices,
                fanout,
                output_device=self.output_device,
                replace=self.replace,
                prob=self.prob,
                exclude_edges=exclude_eids,
            )

            # Get reached nodes.
            curr_reached = defaultdict(list)
            for c_etype in frontier.canonical_etypes:
                (src_type, _, _) = c_etype
                src, _ = frontier.edges(etype=c_etype)
                curr_reached[src_type].append(src)

            # De-duplication.
            curr_reached = {
                ntype: torch.unique(torch.cat(srcs))
                for ntype, srcs in curr_reached.items()
            }

            # Generate type sampling probabilties.
            type_count = {
                node_type: indices.shape[0]
                for node_type, indices in curr_reached.items()
            }
            total_count = sum(type_count.values())
            probs = {
                node_type: count / total_count
                for node_type, count in type_count.items()
            }

            # Upsample rare node types.
            if self.upsample_rare_types:

                # Take scaled square root of probabilities.
                prob_dist = list(probs.values())
                prob_dist = np.sqrt(prob_dist)
                prob_dist = prob_dist / prob_dist.sum()

                # Update probabilities.
                probs = {
                    node_type: prob_dist[i]
                    for i, node_type in enumerate(probs.keys())
                }

            # Generate node counts per type.
            n_per_type = {
                node_type: int(self.fixed_k * prob)
                for node_type, prob in probs.items()
            }
            remainder = self.fixed_k - sum(n_per_type.values())
            for _ in range(remainder):
                node_type = np.random.choice(
                    list(probs.keys()), p=list(probs.values())
                )
                n_per_type[node_type] += 1

            # Downsample nodes.
            curr_reached_k = {}
            for node_type, node_ids in curr_reached.items():

                # Get number of total nodes and number to sample.
                num_nodes = node_ids.shape[0]
                n_to_sample = min(num_nodes, n_per_type[node_type])

                # Downsample nodes of current type.
                random_indices = torch.randperm(num_nodes)[:n_to_sample]
                curr_reached_k[node_type] = node_ids[random_indices]

            # Update seed nodes.
            indices = curr_reached_k
            all_reached_nodes.append(curr_reached_k)

        # Merge all reached nodes before sending to `DGLGraph.subgraph`.
        merged_nodes = {}
        for ntype in g.ntypes:
            merged_nodes[ntype] = torch.unique(
                torch.cat(
                    [reached.get(ntype, []) for reached in all_reached_nodes]
                )
            )
        subg = g.subgraph(
            merged_nodes, relabel_nodes=True, output_device=self.output_device
        )

        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return indices, output_nodes, subg
