"""Negative samplers"""
from collections.abc import Mapping

from .. import backend as F


class _BaseNegativeSampler(object):
    def _generate(self, g, eids, canonical_etype):
        raise NotImplementedError

    def __call__(self, g, eids):
        """Returns negative samples.

        Parameters
        ----------
        g : DGLGraph
            The graph.
        eids : Tensor or dict[etype, Tensor]
            The sampled edges in the minibatch.

        Returns
        -------
        tuple[Tensor, Tensor] or dict[etype, tuple[Tensor, Tensor]]
            The returned source-destination pairs as negative samples.
        """
        if isinstance(eids, Mapping):
            eids = {g.to_canonical_etype(k): v for k, v in eids.items()}
            neg_pair = {k: self._generate(g, v, k) for k, v in eids.items()}
        else:
            assert (
                len(g.canonical_etypes) == 1
            ), "please specify a dict of etypes and ids for graphs with multiple edge types"
            neg_pair = self._generate(g, eids, g.canonical_etypes[0])

        return neg_pair


class PerSourceUniform(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative destination nodes
    for each source node according to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates
    :attr:`k` pairs of negative edges ``(u, v')``, where ``v'`` is chosen
    uniformly from all the nodes of type ``dsttype``.  The resulting edges will
    also have type ``(srctype, etype, dsttype)``.

    Parameters
    ----------
    k : int
        The number of negative samples per edge.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.PerSourceUniform(2)
    >>> neg_sampler(g, torch.tensor([0, 1]))
    (tensor([0, 0, 1, 1]), tensor([1, 0, 2, 3]))
    """

    def __init__(self, k):
        self.k = k

    def _generate(self, g, eids, canonical_etype):
        _, _, vtype = canonical_etype
        shape = F.shape(eids)
        dtype = F.dtype(eids)
        ctx = F.context(eids)
        shape = (shape[0] * self.k,)
        src, _ = g.find_edges(eids, etype=canonical_etype)
        src = F.repeat(src, self.k, 0)
        dst = F.randint(shape, dtype, ctx, 0, g.num_nodes(vtype))
        return src, dst


# Alias
Uniform = PerSourceUniform


class GlobalUniform(_BaseNegativeSampler):
    """Negative sampler that randomly chooses negative source-destination pairs according
    to a uniform distribution.

    For each edge ``(u, v)`` of type ``(srctype, etype, dsttype)``, DGL generates at most
    :attr:`k` pairs of negative edges ``(u', v')``, where ``u'`` is chosen uniformly from
    all the nodes of type ``srctype`` and ``v'`` is chosen uniformly from all the nodes
    of type ``dsttype``.  The resulting edges will also have type
    ``(srctype, etype, dsttype)``.  DGL guarantees that the sampled pairs will not have
    edges in between.

    Parameters
    ----------
    k : int
        The desired number of negative samples to generate per edge.
    exclude_self_loops : bool, optional
        Whether to exclude self-loops from negative samples.  (Default: True)
    replace : bool, optional
        Whether to sample with replacement.  Setting it to True will make things
        faster.  (Default: False)

    Notes
    -----
    This negative sampler will try to generate as many negative samples as possible, but
    it may rarely return less than :attr:`k` negative samples per edge.
    This is more likely to happen if a graph is so small or dense that not many unique
    negative samples exist.

    Examples
    --------
    >>> g = dgl.graph(([0, 1, 2], [1, 2, 3]))
    >>> neg_sampler = dgl.dataloading.negative_sampler.GlobalUniform(2, True)
    >>> neg_sampler(g, torch.LongTensor([0, 1]))
    (tensor([0, 1, 3, 2]), tensor([2, 0, 2, 1]))
    """

    def __init__(self, k, exclude_self_loops=True, replace=False):
        self.k = k
        self.exclude_self_loops = exclude_self_loops
        self.replace = replace

    def _generate(self, g, eids, canonical_etype):
        return g.global_uniform_negative_sampling(
            len(eids) * self.k,
            self.exclude_self_loops,
            self.replace,
            canonical_etype,
        )
