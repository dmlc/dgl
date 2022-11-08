"""GraphSAINT samplers."""
from ..base import DGLError
from ..random import choice
from ..sampling import pack_traces, random_walk
from .base import Sampler, set_edge_lazy_features, set_node_lazy_features

try:
    import torch
except ImportError:
    pass


class SAINTSampler(Sampler):
    """Random node/edge/walk sampler from
    `GraphSAINT: Graph Sampling Based Inductive Learning Method
    <https://arxiv.org/abs/1907.04931>`__

    For each call, the sampler samples a node subset and then returns a node induced subgraph.
    There are three options for sampling node subsets:

    - For :attr:`'node'` sampler, the probability to sample a node is in proportion
      to its out-degree.
    - The :attr:`'edge'` sampler first samples an edge subset and then use the
      end nodes of the edges.
    - The :attr:`'walk'` sampler uses the nodes visited by random walks. It uniformly selects
      a number of root nodes and then performs a fixed-length random walk from each root node.

    Parameters
    ----------
    mode : str
        The sampler to use, which can be :attr:`'node'`, :attr:`'edge'`, or :attr:`'walk'`.
    budget : int or tuple[int]
        Sampler configuration.

        - For :attr:`'node'` sampler, budget specifies the number of nodes
          in each sampled subgraph.
        - For :attr:`'edge'` sampler, budget specifies the number of edges
          to sample for inducing a subgraph.
        - For :attr:`'walk'` sampler, budget is a tuple. budget[0] specifies
          the number of root nodes to generate random walks. budget[1] specifies
          the length of a random walk.

    cache : bool, optional
        If False, it will not cache the probability arrays for sampling. Setting
        it to False is required if you want to use the sampler across different graphs.
    prefetch_ndata : list[str], optional
        The node data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    prefetch_edata : list[str], optional
        The edge data to prefetch for the subgraph.

        See :ref:`guide-minibatch-prefetching` for a detailed explanation of prefetching.
    output_device : device, optional
        The device of the output subgraphs.

    Examples
    --------

    >>> import torch
    >>> from dgl.dataloading import SAINTSampler, DataLoader
    >>> num_iters = 1000
    >>> sampler = SAINTSampler(mode='node', budget=6000)
    >>> # Assume g.ndata['feat'] and g.ndata['label'] hold node features and labels
    >>> dataloader = DataLoader(g, torch.arange(num_iters), sampler, num_workers=4)
    >>> for subg in dataloader:
    ...     train_on(subg)
    """

    def __init__(
        self,
        mode,
        budget,
        cache=True,
        prefetch_ndata=None,
        prefetch_edata=None,
        output_device="cpu",
    ):
        super().__init__()
        self.budget = budget
        if mode == "node":
            self.sampler = self.node_sampler
        elif mode == "edge":
            self.sampler = self.edge_sampler
        elif mode == "walk":
            self.sampler = self.walk_sampler
        else:
            raise DGLError(
                f"Expect mode to be 'node', 'edge' or 'walk', got {mode}."
            )

        self.cache = cache
        self.prob = None
        self.prefetch_ndata = prefetch_ndata or []
        self.prefetch_edata = prefetch_edata or []
        self.output_device = output_device

    def node_sampler(self, g):
        """Node ID sampler for random node sampler"""
        # Alternatively, this can be realized by uniformly sampling an edge subset,
        # and then take the src node of the sampled edges. However, the number of edges
        # is typically much larger than the number of nodes.
        if self.cache and self.prob is not None:
            prob = self.prob
        else:
            prob = g.out_degrees().float().clamp(min=1)
            if self.cache:
                self.prob = prob
        return (
            torch.multinomial(prob, num_samples=self.budget, replacement=True)
            .unique()
            .type(g.idtype)
        )

    def edge_sampler(self, g):
        """Node ID sampler for random edge sampler"""
        src, dst = g.edges()
        if self.cache and self.prob is not None:
            prob = self.prob
        else:
            in_deg = g.in_degrees().float().clamp(min=1)
            out_deg = g.out_degrees().float().clamp(min=1)
            # We can reduce the sample space by half if graphs are always symmetric.
            prob = 1.0 / in_deg[dst.long()] + 1.0 / out_deg[src.long()]
            prob /= prob.sum()
            if self.cache:
                self.prob = prob
        sampled_edges = torch.unique(
            choice(len(prob), size=self.budget, prob=prob)
        )
        sampled_nodes = torch.cat([src[sampled_edges], dst[sampled_edges]])
        return sampled_nodes.unique().type(g.idtype)

    def walk_sampler(self, g):
        """Node ID sampler for random walk sampler"""
        num_roots, walk_length = self.budget
        sampled_roots = torch.randint(0, g.num_nodes(), (num_roots,))
        traces, types = random_walk(g, nodes=sampled_roots, length=walk_length)
        sampled_nodes, _, _, _ = pack_traces(traces, types)
        return sampled_nodes.unique().type(g.idtype)

    def sample(self, g, indices):
        """Sampling function

        Parameters
        ----------
        g : DGLGraph
            The graph to sample from.
        indices : Tensor
            Placeholder not used.

        Returns
        -------
        DGLGraph
            The sampled subgraph.
        """
        node_ids = self.sampler(g)
        sg = g.subgraph(
            node_ids, relabel_nodes=True, output_device=self.output_device
        )
        set_node_lazy_features(sg, self.prefetch_ndata)
        set_edge_lazy_features(sg, self.prefetch_edata)
        return sg
