"""ShaDow-GNN subgraph samplers."""
from .. import transforms
from ..base import NID
from ..sampling.utils import EidExcluder
from .base import Sampler, set_edge_lazy_features, set_node_lazy_features


class ShaDowKHopSampler(Sampler):
    """K-hop subgraph sampler from `Deep Graph Neural Networks with Shallow
    Subgraph Samplers <https://arxiv.org/abs/2012.01380>`__.

    It performs node-wise neighbor sampling and returns the subgraph induced by
    all the sampled nodes. The seed nodes from which the neighbors are sampled
    will appear the first in the induced nodes of the subgraph.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default True
        Whether to sample with replacement
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``. The feature must be
        a scalar on each edge.

    Examples
    --------
    **Node classification**

    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> g = dgl.data.CoraFullDataset()[0]
    >>> sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.DataLoader(
    ...     g, torch.arange(g.num_nodes()), sampler,
    ...     batch_size=5, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, subgraph in dataloader:
    ...     print(subgraph)
    ...     assert torch.equal(input_nodes, subgraph.ndata[dgl.NID])
    ...     assert torch.equal(input_nodes[:output_nodes.shape[0]], output_nodes)
    ...     break
    Graph(num_nodes=529, num_edges=3796,
          ndata_schemes={'label': Scheme(shape=(), dtype=torch.int64),
                         'feat': Scheme(shape=(8710,), dtype=torch.float32),
                         '_ID': Scheme(shape=(), dtype=torch.int64)}
          edata_schemes={'_ID': Scheme(shape=(), dtype=torch.int64)})

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts. Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.ShaDowKHopSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform neighbor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.ShaDowKHopSampler([5, 10, 15], prob='p')
    """

    def __init__(
        self,
        fanouts,
        replace=False,
        prob=None,
        prefetch_node_feats=None,
        prefetch_edge_feats=None,
        output_device=None,
    ):
        super().__init__()
        self.fanouts = fanouts
        self.replace = replace
        self.prob = prob
        self.prefetch_node_feats = prefetch_node_feats
        self.prefetch_edge_feats = prefetch_edge_feats
        self.output_device = output_device

    def sample(
        self, g, seed_nodes, exclude_eids=None
    ):  # pylint: disable=arguments-differ
        """Sampling function.

        Parameters
        ----------
        g : DGLGraph
            The graph to sample nodes from.
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
        output_nodes = seed_nodes
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes,
                fanout,
                output_device=self.output_device,
                replace=self.replace,
                prob=self.prob,
                exclude_edges=exclude_eids,
            )
            block = transforms.to_block(frontier, seed_nodes)
            seed_nodes = block.srcdata[NID]

        subg = g.subgraph(
            seed_nodes, relabel_nodes=True, output_device=self.output_device
        )
        if exclude_eids is not None:
            subg = EidExcluder(exclude_eids)(subg)

        set_node_lazy_features(subg, self.prefetch_node_feats)
        set_edge_lazy_features(subg, self.prefetch_edge_feats)

        return seed_nodes, output_nodes, subg
