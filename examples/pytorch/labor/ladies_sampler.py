# referenced the following implementation: https://github.com/BarclayII/dgl/blob/ladies/examples/pytorch/ladies/ladies2.py

import dgl
import dgl.function as fn
import torch


def find_indices_in(a, b):
    b_sorted, indices = torch.sort(b)
    sorted_indices = torch.searchsorted(b_sorted, a)
    sorted_indices[sorted_indices >= indices.shape[0]] = 0
    return indices[sorted_indices]


def union(*arrays):
    return torch.unique(torch.cat(arrays))


def normalized_edata(g, weight=None):
    with g.local_scope():
        if weight is None:
            weight = "W"
            g.edata[weight] = torch.ones(g.number_of_edges(), device=g.device)
        g.update_all(fn.copy_e(weight, weight), fn.sum(weight, "v"))
        g.apply_edges(lambda edges: {"w": 1 / edges.dst["v"]})
        return g.edata["w"]


class LadiesSampler(dgl.dataloading.BlockSampler):
    def __init__(
        self,
        nodes_per_layer,
        importance_sampling=True,
        weight="w",
        out_weight="edge_weights",
        replace=False,
    ):
        super().__init__()
        self.nodes_per_layer = nodes_per_layer
        self.importance_sampling = importance_sampling
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace

    def compute_prob(self, g, seed_nodes, weight, num):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges
        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
        insg = dgl.in_subgraph(g, seed_nodes)
        insg = dgl.compact_graphs(insg, seed_nodes)
        if self.importance_sampling:
            out_frontier = dgl.reverse(insg, copy_edata=True)
            weight = weight[out_frontier.edata[dgl.EID].long()]
            prob = dgl.ops.copy_e_sum(out_frontier, weight**2)
            # prob = torch.sqrt(prob)
        else:
            prob = torch.ones(insg.num_nodes())
            prob[insg.out_degrees() == 0] = 0
        return prob, insg

    def select_neighbors(self, prob, num):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample
        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        neighbor_nodes_idx = torch.multinomial(
            prob, min(num, prob.shape[0]), replacement=self.replace
        )
        return neighbor_nodes_idx

    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes, P_sg, W_sg):
        """
        insg : the subgraph yielded by compute_prob()
        neighbor_nodes_idx : the sampled nodes from the subgraph @insg, yielded by select_neighbors()
        seed_nodes_local_idx : the indices of seed nodes in the selected neighbor nodes, also yielded
                               by select_neighbors()
        P_sg : unnormalized probability of each node being sampled, yielded by compute_prob()
        W_sg : edge weights of @insg
        return : the block.
        """
        seed_nodes_idx = find_indices_in(seed_nodes, insg.ndata[dgl.NID])
        u_nodes = union(neighbor_nodes_idx, seed_nodes_idx)
        sg = insg.subgraph(u_nodes.type(insg.idtype))
        u, v = sg.edges()
        lu = sg.ndata[dgl.NID][u.long()]
        s = find_indices_in(lu, neighbor_nodes_idx)
        eg = dgl.edge_subgraph(
            sg, lu == neighbor_nodes_idx[s], relabel_nodes=False
        )
        eg.ndata[dgl.NID] = sg.ndata[dgl.NID][: eg.num_nodes()]
        eg.edata[dgl.EID] = sg.edata[dgl.EID][eg.edata[dgl.EID].long()]
        sg = eg
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID].long()]
        P = P_sg[u_nodes.long()]
        W = W_sg[sg.edata[dgl.EID].long()]
        W_tilde = dgl.ops.e_div_u(sg, W, P)
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)
        d = sg.in_degrees()
        W_tilde = dgl.ops.e_mul_v(sg, W_tilde, d / W_tilde_sum)

        block = dgl.to_block(sg, seed_nodes_idx.type(sg.idtype))
        block.edata[self.output_weight] = W_tilde
        # correct node ID mapping
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID].long()]
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID].long()]

        sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID].long()]
        block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID].long()]
        return block

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for block_id in reversed(range(len(self.nodes_per_layer))):
            num_nodes_to_sample = self.nodes_per_layer[block_id]
            W = g.edata[self.edge_weight]
            prob, insg = self.compute_prob(
                g, seed_nodes, W, num_nodes_to_sample
            )
            neighbor_nodes_idx = self.select_neighbors(
                prob, num_nodes_to_sample
            )
            block = self.generate_block(
                insg,
                neighbor_nodes_idx.type(g.idtype),
                seed_nodes.type(g.idtype),
                prob,
                W[insg.edata[dgl.EID].long()],
            )
            seed_nodes = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks


class PoissonLadiesSampler(LadiesSampler):
    def __init__(
        self,
        nodes_per_layer,
        importance_sampling=True,
        weight="w",
        out_weight="edge_weights",
        skip=False,
    ):
        super().__init__(
            nodes_per_layer, importance_sampling, weight, out_weight
        )
        self.eps = 0.9999
        self.skip = skip

    def compute_prob(self, g, seed_nodes, weight, num):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges
        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
        prob, insg = super().compute_prob(g, seed_nodes, weight, num)

        one = torch.ones_like(prob)
        if prob.shape[0] <= num:
            return one, insg

        c = 1.0
        for i in range(50):
            S = torch.sum(torch.minimum(prob * c, one).to(torch.float64)).item()
            if min(S, num) / max(S, num) >= self.eps:
                break
            else:
                c *= num / S

        if self.skip:
            skip_nodes = find_indices_in(seed_nodes, insg.ndata[dgl.NID])
            prob[skip_nodes] = float("inf")

        return torch.minimum(prob * c, one), insg

    def select_neighbors(self, prob, num):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample
        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        neighbor_nodes_idx = torch.arange(prob.shape[0], device=prob.device)[
            torch.bernoulli(prob) == 1
        ]
        return neighbor_nodes_idx
