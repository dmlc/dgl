import torch as th
from torch.autograd import Function

"""
Non-DGL GraphSage Model Utils
"""
def construct_mask(mask_length: th.Tensor, max_length):
    # mask_length = mask_length.squeeze(-1).cpu()
    mask_length = mask_length.cpu()
    # max_length = th.max(mask_length)
    mask = th.ones(*mask_length.shape, max_length)
    mask[th.arange(mask_length.shape[0]), mask_length - 1] = 1
    return th.cumprod(mask, dim=1)


def masked_softmax(vector: th.Tensor,
                   mask: th.Tensor,
                   dim: int = -1,
                   memory_efficient: bool = False,
                   mask_fill_value: float = -1e32) -> th.Tensor:
    """
    ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular softmax.

    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
    masked positions so that the probabilities of those positions would be approximately 0.
    This is not accurate in math, but works for most cases and consumes less memory.

    In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
    returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
    a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
    will treat every element as equal, and do softmax over equal numbers.
    """
    if mask is None:
        result = th.nn.functional.softmax(vector, dim=dim)
    else:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        if not memory_efficient:
            # To limit numerical errors from large vector elements outside the mask, we zero these out.
            result = th.nn.functional.softmax(vector * mask, dim=dim)
            result = result * mask
            result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
        else:
            masked_vector = vector.masked_fill((1 - mask).byte(), mask_fill_value)
            result = th.nn.functional.softmax(masked_vector, dim=dim)
    return result


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

class BatchedTrace(Function):
    @staticmethod
    def forward(ctx, tensor):
        ctx.save_for_backward(tensor)
        output = th.tensor([th.trace(tensor[i,...]) for i in
                               range(tensor.shape[0])]).unsqueeze(0).t()
        return output
    @staticmethod
    def backward(ctx, grad_output):
        tensor = ctx.saved_tensors[0]
        output_list = []
        output = th.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            mat = (th.eye(tensor.shape[1])*grad_output[i]).unsqueeze(0)
            output_list.append(mat)
        output = output + th.cat(output_list, 0)

        return output



