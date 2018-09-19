from __future__ import absolute_import

from ._ffi.function import _init_api
from . import backend as F
from . import utils

class GraphIndex(object):
    """Graph index object.

    Parameters
    ----------
    graph_data : graph data
        Data to initialize graph. Same as networkx's semantics.
    """
    def __init__(self, graph_data=None):
        # TODO: convert from graph data
        self._handle = _CAPI_DGLGraphCreate()

    def __del__(self):
        """Free this graph index object."""
        _CAPI_DGLGraphFree(self._handle)

    def add_nodes(self, num):
        """Add nodes.
        
        Parameters
        ----------
        num : int
            Number of nodes to be added.
        """
        _CAPI_DGLGraphAddVertices(self._handle, num);

    def add_edge(self, u, v):
        """Add one edge.
        
        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.
        """
        _CAPI_DGLGraphAddEdge(self._handle, u, v);

    def add_edges(self, u, v):
        """Add many edges.
        
        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.
        """
        #u = utils.Index(u)
        #v = utils.Index(v)
        #u_array = F.asdglarray(u.totensor())
        #v_array = F.asdglarray(v.totensor())
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        _CAPI_DGLGraphAddEdges(self._handle, u_array, v_array)

    def clear(self):
        """Clear the graph."""
        _CAPI_DGLGraphClear(self._handle)

    def number_of_nodes(self):
        """Return the number of nodes.

        Returns
        -------
        int
            The number of nodes
        """
        return _CAPI_DGLGraphNumVertices(self._handle)

    def number_of_edges(self):
        """Return the number of edges.

        Returns
        -------
        int
            The number of edges
        """
        return _CAPI_DGLGraphNumEdges(self._handle)

    def has_node(self, vid):
        """Return true if the node exists.

        Parameters
        ----------
        vid : int
            The nodes

        Returns
        -------
        bool
            True if the node exists
        """
        return _CAPI_DGLGraphHasVertex(self._handle, vid)

    def has_nodes(self, vids):
        """Return true if the nodes exist.

        Parameters
        ----------
        vid : utils.Index
            The nodes

        Returns
        -------
        utils.Index
            0-1 array indicating existence
        """
        vid_array = vids.todgltensor()
        return utils.Index(_CAPI_DGLGraphHasVertices(self._handle, vid_array))

    def has_edge(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        bool
            True if the edge exists
        """
        return _CAPI_DGLGraphHasEdge(self._handle, u, v)

    def has_edges(self, u, v):
        """Return true if the edge exists.

        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.

        Returns
        -------
        utils.Index
            0-1 array indicating existence
        """
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        return utils.Index(_CAPI_DGLGraphHasEdges(self._handle, u_array, v_array))

    def predecessors(self, v, radius=1):
        """Return the predecessors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        utils.Index
            Array of predecessors
        """
        return utils.Index(_CAPI_DGLGraphPredecessors(self._handle, v, radius))

    def successors(self, v, radius=1):
        """Return the successors of the node.

        Parameters
        ----------
        v : int
            The node.
        radius : int, optional
            The radius of the neighborhood.

        Returns
        -------
        utils.Index
            Array of successors
        """
        return utils.Index(_CAPI_DGLGraphSuccessors(self._handle, v, radius))

    def edge_id(self, u, v):
        """Return the id of the edge.

        Parameters
        ----------
        u : int
            The src node.
        v : int
            The dst node.

        Returns
        -------
        int
            The edge id.
        """
        return _CAPI_DGLGraphEdgeId(self._handle, u, v)

    def edge_ids(self, u, v):
        """Return the edge ids.

        Parameters
        ----------
        u : utils.Index
            The src nodes.
        v : utils.Index
            The dst nodes.

        Returns
        -------
        utils.Index
            Teh edge id array.
        """
        u_array = u.todgltensor()
        v_array = v.todgltensor()
        return utils.Index(_CAPI_DGLGraphEdgeIds(self._handle, u_array, v_array))

    def in_edges(self, v):
        """Return the in edges of the node(s).

        Parameters
        ----------
        v : utils.Index
            The node(s).
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        if len(v) == 1:
            edge_array = _CAPI_DGLGraphInEdges_1(self._handle, v[0])
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphInEdges_2(self._handle, v_array)
        src = utils.Index(edge_array(0))
        dst = utils.Index(edge_array(1))
        eid = utils.Index(edge_array(2))
        return src, dst, eid

    def out_edges(self, v):
        """Return the out edges of the node(s).

        Parameters
        ----------
        v : utils.Index
            The node(s).
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        if len(v) == 1:
            edge_array = _CAPI_DGLGraphOutEdges_1(self._handle, v[0])
        else:
            v_array = v.todgltensor()
            edge_array = _CAPI_DGLGraphOutEdges_2(self._handle, v_array)
        src = utils.Index(edge_array(0))
        dst = utils.Index(edge_array(1))
        eid = utils.Index(edge_array(2))
        return src, dst, eid

    def edges(self, sorted=False):
        """Return all the edges

        Parameters
        ----------
        sorted : bool
            True if the returned edges are sorted by their ids.
        
        Returns
        -------
        utils.Index
            The src nodes.
        utils.Index
            The dst nodes.
        utils.Index
            The edge ids.
        """
        edge_array = _CAPI_DGLGraphEdges(self._handle, sorted)
        src = edge_array(0)
        dst = edge_array(1)
        eid = edge_array(2)
        return src, dst, eid

    def in_degree(self, v):
        """Return the in degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The in degree.
        """
        return _CAPI_DGLGraphInDegree(self._handle, v)

    def in_degrees(self, v):
        """Return the in degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The in degree array.
        """
        v_array = v.todgltensor()
        return utils.Index(_CAPI_DGLGraphInDegrees(self._handle, v_array))

    def out_degree(self, v):
        """Return the out degree of the node.

        Parameters
        ----------
        v : int
            The node.

        Returns
        -------
        int
            The out degree.
        """
        return _CAPI_DGLGraphOutDegree(self._handle, v)

    def out_degrees(self, v):
        """Return the out degrees of the nodes.

        Parameters
        ----------
        v : utils.Index
            The nodes.

        Returns
        -------
        int
            The out degree array.
        """
        v_array = v.todgltensor()
        return utils.Index(_CAPI_DGLGraphOutDegrees(self._handle, v_array))

    def asnetworkx(self):
        """Convert to networkx graph.

        Returns
        -------
        networkx.DiGraph
            The nx graph
        """
        # TODO
        return None

_init_api("dgl.graph")
