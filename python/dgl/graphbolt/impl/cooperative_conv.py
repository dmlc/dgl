"""Graphbolt cooperative convolution."""
import torch

from ..sampled_subgraph import SampledSubgraph
from ..subgraph_sampler import all_to_all

__all__ = ["CooperativeConvFunction", "CooperativeConv"]


class CooperativeConvFunction(torch.autograd.Function):
    """Cooperative convolution operation from Cooperative Minibatching.

    Implements the `all-to-all` message passing algorithm
    in Cooperative Minibatching, which was initially proposed in
    `Deep Graph Library PR#4337<https://github.com/dmlc/dgl/pull/4337>`__ and
    was later first fully described in
    `Cooperative Minibatching in Graph Neural Networks
    <https://arxiv.org/abs/2310.12403>`__.
    Cooperation between the GPUs eliminates duplicate work performed across the
    GPUs due to the overlapping sampled k-hop neighborhoods of seed nodes when
    performing GNN minibatching. This reduces the redundant computations across
    GPUs at the expense of communication.
    """

    @staticmethod
    def forward(ctx, subgraph: SampledSubgraph, h: torch.Tensor):
        counts_sent = subgraph._counts_sent
        counts_received = subgraph._counts_received
        seed_inverse_ids = subgraph._seed_inverse_ids
        seed_sizes = subgraph._seed_sizes
        ctx.save_for_backward(
            counts_sent, counts_received, seed_inverse_ids, seed_sizes
        )
        out = h.new_empty((sum(counts_sent),) + h.shape[1:])
        all_to_all(
            torch.split(out, counts_sent),
            torch.split(h[seed_inverse_ids], counts_received),
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (
            counts_sent,
            counts_received,
            seed_inverse_ids,
            seed_sizes,
        ) = ctx.saved_tensors
        out = grad_output.new_empty(
            (sum(counts_received),) + grad_output.shape[1:]
        )
        all_to_all(
            torch.split(out, counts_received),
            torch.split(grad_output, counts_sent),
        )
        i = out.new_empty(2, out.shape[0], dtype=torch.int64)
        i[0] = torch.arange(out.shape[0], device=grad_output.device)  # src
        i[1] = seed_inverse_ids  # dst
        coo = torch.sparse_coo_tensor(i, 1, size=(seed_sizes, i.shape[1]))
        rout = torch.sparse.mm(coo, out)
        return None, rout


class CooperativeConv(torch.nn.Module):
    """Cooperative convolution operation from Cooperative Minibatching.

    Implements the `all-to-all` message passing algorithm
    in Cooperative Minibatching, which was initially proposed in
    `Deep Graph Library PR#4337<https://github.com/dmlc/dgl/pull/4337>`__ and
    was later first fully described in
    `Cooperative Minibatching in Graph Neural Networks
    <https://arxiv.org/abs/2310.12403>`__.
    Cooperation between the GPUs eliminates duplicate work performed across the
    GPUs due to the overlapping sampled k-hop neighborhoods of seed nodes when
    performing GNN minibatching. This reduces the redundant computations across
    GPUs at the expense of communication.
    """

    def __init__(self):
        super().__init__()

    def forward(self, subgraph: SampledSubgraph, x: torch.Tensor):
        return CooperativeConvFunction.apply(subgraph, x)
