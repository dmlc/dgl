"""ShaDow-GNN subgraph samplers."""
from .. import backend as F
from .. import transform
from ..utils import prepare_tensor_or_dict
from ..base import NID
from ..sampling import sample_neighbors
from .neighbor import NeighborSamplingMixin
from .dataloader import exclude_edges, Sampler

class ShaDowKHopSampler(NeighborSamplingMixin, Sampler):
    """K-hop subgraph sampler used by
    `ShaDow-GNN <https://openreview.net/forum?id=GIeGTl8EYx>`__.

    It performs node-wise neighbor sampling but instead of returning a list of
    MFGs, it returns a single subgraph induced by all the sampled nodes.

    This is used in conjunction with :class:`dgl.dataloading.pytorch.NodeDataLoader`
    and :class:`dgl.dataloading.pytorch.EdgeDataLoader`.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int] or None]
        List of neighbors to sample per edge type for each GNN layer, starting from the
        first layer.

        If the graph is homogeneous, only an integer is needed for each layer.

        If None is provided for one layer, all neighbors will be included regardless of
        edge types.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default True
        Whether to sample with replacement
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature with the given name.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, (subgraph,) in dataloader:
    ...     train_on(subgraph)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.ShaDowKHopSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)
    """
    def __init__(self, fanouts, replace=False, prob=None, output_ctx=None):
        super().__init__(output_ctx)
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.set_output_context(output_ctx)

    def sample(self, g, seeds, exclude_eids=None):
        self._build_fanout(len(self.fanouts), g)
        self._build_prob_arrays(g)
        seeds = prepare_tensor_or_dict(g, seeds, 'seed nodes')
        output_nodes = seeds

        for i in range(len(self.fanouts)):
            fanout = self.fanouts[i]
            frontier = sample_neighbors(
                g, seeds, fanout, replace=self.replace, prob=self.prob_arrays)
            block = transform.to_block(frontier, seeds)
            seeds = block.srcdata[NID]

        sg = g.subgraph(seeds, relabel_nodes=True)
        sg = exclude_edges(sg, exclude_eids, self.output_device)

        return seeds, output_nodes, [sg]
