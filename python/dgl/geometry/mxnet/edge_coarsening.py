"""Farthest Point Sampler for mxnet Geometry package"""
#pylint: disable=no-member, invalid-name

from mxnet import nd
from mxnet.gluon import nn

from ..capi import edge_coarsening

class EdgeCoarsening(nn.Block):
    r"""
    Description
    -----------
    The edge coarsening procedure used in
    `Metis <http://cacs.usc.edu/education/cs653/Karypis-METIS-SIAMJSC98.pdf>`__
    and
    `Graclus <https://www.cs.utexas.edu/users/inderjit/public_papers/multilevel_pami.pdf>`__
    for homogeneous graph coarsening. This procedure keeps picking an unmarked
    vertex and matching it with one its unmarked neighbors (that maximizes its
    edge weight) until no match can be done.

    If no edge weight is given, this procedure will randomly pick neighbor for each
    vertex.

    The GPU implementation is based on `A GPU Algorithm for Greedy Graph Matching
    <http://www.staff.science.uu.nl/~bisse101/Articles/match12.pdf>`__
    """
    def __init__(self):
        super(EdgeCoarsening, self).__init__()

    def forward(self, graph, edge_weights=None, relabel_idx=True):
        r"""
        Description
        -----------
        Perform edge coarsening

        NOTE: The input graph must be bi-directed (undirected) graph. Call :obj:`dgl.to_bidirected`
        if you are not sure your graph is bi-directed.

        Parameters
        ----------
        graph : DGLGraph
            The input homogeneous graph.
        edge_weight : mxnet.NDArray, optional
            The edge weight tensor holding non-negative scalar weight for each edge.
            default: :obj:`None`
        relabel_idx : bool, optional
            If true, relabel resulting node labels to have consecutive node ids.
            default: :obj:`True`

        Returns
        -------
        a 1-D tensor
        A vector with each element that indicates the cluster ID of a vertex.
        """
        return edge_coarsening(graph, edge_weights=edge_weights, relabel_idx=relabel_idx)
