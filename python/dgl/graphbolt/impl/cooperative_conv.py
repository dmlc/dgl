"""Graphbolt cooperative convolution."""
import torch

from ..sampled_subgraph import SampledSubgraph
from ..subgraph_sampler import all_to_all, convert_to_hetero, revert_to_homo

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
    def forward(ctx, subgraph: SampledSubgraph, tensor: torch.Tensor):
        """Implements the forward pass."""
        counts_sent = convert_to_hetero(subgraph._counts_sent)
        counts_received = convert_to_hetero(subgraph._counts_received)
        seed_inverse_ids = convert_to_hetero(subgraph._seed_inverse_ids)
        seed_sizes = convert_to_hetero(subgraph._seed_sizes)
        ctx.save_for_backward(
            counts_sent, counts_received, seed_inverse_ids, seed_sizes
        )
        outs = {}
        for ntype, tensor in convert_to_hetero(tensor).items():
            out = tensor.new_empty(
                (sum(counts_sent[ntype]),) + tensor.shape[1:]
            )
            all_to_all(
                torch.split(out, counts_sent[ntype]),
                torch.split(
                    tensor[seed_inverse_ids[ntype]], counts_received[ntype]
                ),
            )
            outs[ntype] = out
        return revert_to_homo(out)

    @staticmethod
    def backward(ctx, grad_output):
        """Implements the forward pass."""
        (
            counts_sent,
            counts_received,
            seed_inverse_ids,
            seed_sizes,
        ) = ctx.saved_tensors
        outs = {}
        for ntype, grad_output in convert_to_hetero(grad_output).items():
            out = grad_output.new_empty(
                (sum(counts_received[ntype]),) + grad_output.shape[1:]
            )
            all_to_all(
                torch.split(out, counts_received[ntype]),
                torch.split(grad_output, counts_sent[ntype]),
            )
            i = out.new_empty(2, out.shape[0], dtype=torch.int64)
            i[0] = torch.arange(out.shape[0], device=grad_output.device)  # src
            i[1] = seed_inverse_ids[ntype]  # dst
            coo = torch.sparse_coo_tensor(
                i, 1, size=(seed_sizes[ntype], i.shape[1])
            )
            outs[ntype] = torch.sparse.mm(coo, out)
        return None, revert_to_homo(outs)


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

    def forward(self, subgraph: SampledSubgraph, x: torch.Tensor):
        """Implements the forward pass."""
        return CooperativeConvFunction.apply(subgraph, x)
