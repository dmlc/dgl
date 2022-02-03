"""Data loading components for neighbor sampling"""
from .dataloader import BlockSampler
from .. import sampling, distributed
from .. import ndarray as nd
from .. import backend as F
from ..base import ETYPE

class NeighborSamplingMixin(object):
    """Mixin object containing common optimizing routines that caches fanout and probability
    arrays.

    The mixin requires the object to have the following attributes:

    - :attr:`prob`: The edge feature name that stores the (unnormalized) probability.
    - :attr:`fanouts`: The list of fanouts (either an integer or a dictionary of edge
      types and integers).

    The mixin will generate the following attributes:

    - :attr:`prob_arrays`: List of DGL NDArrays containing the unnormalized probabilities
      for every edge type.
    - :attr:`fanout_arrays`: List of DGL NDArrays containing the fanouts for every edge
      type at every layer.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)       # forward to base classes
        self.fanout_arrays = []
        self.prob_arrays = None

    def _build_prob_arrays(self, g):
        if self.prob is not None:
            self.prob_arrays = [F.to_dgl_nd(g.edges[etype].data[self.prob]) for etype in g.etypes]
        elif self.prob_arrays is None:
            # build prob_arrays only once
            self.prob_arrays = [nd.array([], ctx=nd.cpu())] * len(g.etypes)

    def _build_fanout(self, block_id, g):
        assert not self.fanouts is None, \
            "_build_fanout() should only be called when fanouts is not None"
        # build fanout_arrays only once for each layer
        while block_id >= len(self.fanout_arrays):
            for i in range(len(self.fanouts)):
                fanout = self.fanouts[i]
                if not isinstance(fanout, dict):
                    fanout_array = [int(fanout)] * len(g.etypes)
                else:
                    if len(fanout) != len(g.etypes):
                        raise DGLError('Fan-out must be specified for each edge type '
                                       'if a dict is provided.')
                    fanout_array = [None] * len(g.etypes)
                    for etype, value in fanout.items():
                        fanout_array[g.get_etype_id(etype)] = value
                self.fanout_arrays.append(
                    F.to_dgl_nd(F.tensor(fanout_array, dtype=F.int64)))

class MultiLayerNeighborSampler(NeighborSamplingMixin, BlockSampler):
    """Sampler that builds computational dependency of node representations via
    neighbor sampling for multilayer GNN.

    This sampler will make every node gather messages from a fixed number of neighbors
    per edge type.  The neighbors are picked uniformly.

    Parameters
    ----------
    fanouts : list[int] or list[dict[etype, int]]
        List of neighbors to sample per edge type for each GNN layer, with the i-th
        element being the fanout for the i-th GNN layer.

        If only a single integer is provided, DGL assumes that every edge type
        will have the same fanout.

        If -1 is provided for one edge type on one layer, then all inbound edges
        of that edge type will be included.
    replace : bool, default False
        Whether to sample with replacement
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the MFG.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.
    prob : str, optional
        If given, the probability of each neighbor being sampled is proportional
        to the edge feature value with the given name in ``g.edata``.  The feature must be
        a scalar on each edge.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from 5, 10, 15 neighbors for
    the first, second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 15])
    >>> dataloader = dgl.dataloading.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    If training on a heterogeneous graph and you want different number of neighbors for each
    edge type, one should instead provide a list of dicts.  Each dict would specify the
    number of neighbors to pick per edge type.

    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([
    ...     {('user', 'follows', 'user'): 5,
    ...      ('user', 'plays', 'game'): 4,
    ...      ('game', 'played-by', 'user'): 3}] * 3)

    If you would like non-uniform neighbor sampling:

    >>> g.edata['p'] = torch.rand(g.num_edges())   # any non-negative 1D vector works
    >>> sampler = dgl.dataloading.MultiLayerNeighborSampler([5, 10, 15], prob='p')

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """
    def __init__(self, fanouts, replace=False, return_eids=False, prob=None):
        super().__init__(len(fanouts), return_eids)

        self.fanouts = fanouts
        self.replace = replace

        # used to cache computations and memory allocations
        # list[dgl.nd.NDArray]; each array stores the fan-outs of all edge types
        self.prob = prob

    @classmethod
    def exclude_edges_in_frontier(cls, g):
        return not isinstance(g, distributed.DistGraph) and g.device == F.cpu() \
               and not g.is_pinned()

    def sample_frontier(self, block_id, g, seed_nodes, exclude_eids=None):
        fanout = self.fanouts[block_id]
        if isinstance(g, distributed.DistGraph):
            if len(g.etypes) > 1: # heterogeneous distributed graph
                frontier = distributed.sample_etype_neighbors(
                    g, seed_nodes, ETYPE, fanout, replace=self.replace)
            else:
                frontier = distributed.sample_neighbors(
                    g, seed_nodes, fanout, replace=self.replace)
        else:
            self._build_fanout(block_id, g)
            self._build_prob_arrays(g)

            frontier = sampling.sample_neighbors(
                g, seed_nodes, self.fanout_arrays[block_id],
                replace=self.replace, prob=self.prob_arrays, exclude_edges=exclude_eids)
        return frontier


class MultiLayerFullNeighborSampler(MultiLayerNeighborSampler):
    """Sampler that builds computational dependency of node representations by taking messages
    from all neighbors for multilayer GNN.

    This sampler will make every node gather messages from every single neighbor per edge type.

    Parameters
    ----------
    n_layers : int
        The number of GNN layers to sample.
    return_eids : bool, default False
        Whether to return the edge IDs involved in message passing in the MFG.
        If True, the edge IDs will be stored as an edge feature named ``dgl.EID``.

    Examples
    --------
    To train a 3-layer GNN for node classification on a set of nodes ``train_nid`` on
    a homogeneous graph where each node takes messages from all neighbors for the first,
    second, and third layer respectively (assuming the backend is PyTorch):

    >>> sampler = dgl.dataloading.MultiLayerFullNeighborSampler(3)
    >>> dataloader = dgl.dataloading.NodeDataLoader(
    ...     g, train_nid, sampler,
    ...     batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for input_nodes, output_nodes, blocks in dataloader:
    ...     train_on(blocks)

    Notes
    -----
    For the concept of MFGs, please refer to
    :ref:`User Guide Section 6 <guide-minibatch>` and
    :doc:`Minibatch Training Tutorials <tutorials/large/L0_neighbor_sampling_overview>`.
    """
    def __init__(self, n_layers, return_eids=False):
        super().__init__([-1] * n_layers, return_eids=return_eids)

    @classmethod
    def exclude_edges_in_frontier(cls, g):
        return False
