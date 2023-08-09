"""SpotTarget: Target edge excluder for link prediction"""
import torch

from .base import find_exclude_eids


class SpotTarget(object):
    """Callable excluder object to exclude the edges by the degree threshold.

    Besides excluding all the edges or given edges in the edge sampler
    ``dgl.dataloading.as_edge_prediction_sampler`` in link prediction training,
    this excluder can extend the exclusion function by only excluding the edges incident
    to low-degree nodes in the graph to bring the performance increase in training
    link prediction model. This function will exclude the edge if incident to at least
    one node with degree larger or equal to ``degree_threshold``. The performance
    boost by excluding the target edges incident to low-degree nodes can be found
    in this paper: https://arxiv.org/abs/2306.00899

    Parameters
    ----------
    g : DGLGraph
        The graph.
    exclude : Union[str, callable]
        Whether and how to exclude dependencies related to the sampled edges in the
        minibatch.  Possible values are

        * ``self``, for excluding the edges in the current minibatch.

        * ``reverse_id``, for excluding not only the edges in the current minibatch but
          also their reverse edges according to the ID mapping in the argument
          :attr:`reverse_eids`.

        * ``reverse_types``, for excluding not only the edges in the current minibatch
          but also their reverse edges stored in another type according to
          the argument :attr:`reverse_etypes`.

        * User-defined exclusion rule. It is a callable with edges in the current
          minibatch as a single argument and should return the edges to be excluded.
    degree_threshold : int
        The threshold of node degrees, if the source or target node of an edge incident to
        has larger or equal degrees than ``degree_threshold``, this edge will be excluded from
        the graph
    reverse_eids : Tensor or dict[etype, Tensor], optional
        A tensor of reverse edge ID mapping.  The i-th element indicates the ID of
        the i-th edge's reverse edge.

        If the graph is heterogeneous, this argument requires a dictionary of edge
        types and the reverse edge ID mapping tensors.
    reverse_etypes : dict[etype, etype], optional
        The mapping from the original edge types to their reverse edge types.

    Examples
    --------
    .. code:: python
       low_degree_excluder = SpotTarget(g, degree_threshold=10)
       sampler = as_edge_prediction_sampler(sampler, exclude=low_degree_excluder,
       reverse_eids=reverse_eids, negative_sampler=negative_sampler.Uniform(1))
    """

    def __init__(
        self,
        g,
        exclude,
        degree_threshold=10,
        reverse_eids=None,
        reverse_etypes=None,
    ):
        self.g = g
        self.exclude = exclude
        self.degree_threshold = degree_threshold
        self.reverse_eids = reverse_eids
        self.reverse_etypes = reverse_etypes

    def __call__(self, seed_edges):
        g = self.g
        src, dst = g.find_edges(seed_edges)
        head_degree = g.in_degrees(src)
        tail_degree = g.in_degrees(dst)

        degree = torch.min(head_degree, tail_degree)
        degree_mask = degree < self.degree_threshold
        edges_need_to_exclude = seed_edges[degree_mask]
        return find_exclude_eids(
            g,
            edges_need_to_exclude,
            self.exclude,
            self.reverse_eids,
            self.reverse_etypes,
        )
