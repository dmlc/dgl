import networkx as nx
from networkx.classes.filters import no_filter, show_nodes, show_edges
import dgl
import dgl.backend as F
from dgl.frame import Frame
import dgl.utils as utils

class DGLSubGraph(dgl.DGLGraph):
    # TODO(gaiyu): ReadOnlyGraph
    def __init__(self, graph, filter_node=no_filter, filter_edge=no_filter):
        super(DGLSubGraph, self).__init__()
        nx.graphviews.SubDiGraph.__init__(self, graph, filter_node, filter_edge)
        nid = utils.convert_to_id_tensor(list(self.nodes), self.context)
        self._node_frame = SubFrame(nid, graph._node_frame._columns)
        self._edge_list = list(self.edges)
        u, v = self.cached_graph.edges()
        eid = self.cached_graph.get_edge_id(u, v)
        self._edge_frame = SubFrame(eid, graph._edge_frame._columns)
