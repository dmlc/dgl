"""Class for subgraph data structure."""
from __future__ import absolute_import

from .graph import DGLGraph
from .base import DGLError

DGLSubGraph = DGLGraph

class DGLSubGraph(DGLGraph):
    """The subgraph class.

    There are two subgraph modes: shared and non-shared.

    For the "non-shared" mode, the user needs to explicitly call
    ``copy_from_parent`` to copy node/edge features from its parent graph.
    * If the user tries to get node/edge features before ``copy_from_parent``,
      s/he will get nothing.
    * If the subgraph already has its own node/edge features, ``copy_from_parent``
      will override them.
    * Any update on the subgraph's node/edge features will not be seen
      by the parent graph. As such, the memory consumption is of the order
      of the subgraph size.
    * To write the subgraph's node/edge features back to parent graph. There are two options:
      (1) Use ``copy_to_parent`` API to write node/edge features back.
      (2) [TODO] Use ``dgl.merge`` to merge multiple subgraphs back to one parent.

    Parameters
    ----------
    parent : DGLGraph
        The parent graph
    sgi : SubgraphIndex
        Internal subgraph data structure.
    """
    def __init__(self, parent, sgi, shared=False):
        super(DGLSubGraph, self).__init__(graph_data=sgi.graph,
                                          readonly=True,
                                          parent=parent,
                                          sgi=sgi)
        if shared:
            raise DGLError('Shared mode is not yet supported.')
