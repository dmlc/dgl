import networkx as nx
from networkx.classes.filters import no_filter, show_nodes, show_edges
import dgl
import dgl.backend as F
from dgl.frame import Frame
import dgl.utils as utils

class SubFrame(Frame):
    def __init__(self, rowids, columns):
        """
        Parameters
        ----------
        rowids : F.Tensor
        columns : dict
        """
        self._columns = columns.copy()
        if columns:
            self._num_rows = F.shape(rowids)[0]
        else:
            self._num_rows = 0
        self._rowids = rowids
        self._dirty = {key : False for key in self._columns}

    def _getcol(self, key):
        col = self._columns[key]
        return col if self._dirty[key] else F.gather_row(col, self._rowids)

    def __getitem__(self, key):
        assert type(key) is str, 'Partial query not supported.'
        return self._getcol(key)

    def __setitem__(self, key, val):
        assert type(key) is str, 'Partial update not supported.'
        self._columns[key] = val
        self._dirty[key] = True

    def __delitem__(self, key):
        super(SubFrame, self).__delitem__(key)
        del self._dirty[key]

    def pop(self, key):
        del self._dirty[key]
        return super(SubFrame, self).pop(key)
        
    def add_column(self, name, col):
        assert F.shape(col)[0] == self._num_rows
        self._columns[name] = col

    def append(self, other):
        raise NotImplementedError()

    def select_rows(rowids):
        raise RuntimeError('Partial query not supported.')

    def update_rows(self, rowids, other):
        raise RuntimeError('Partial update not supported.')

    def __iter__(self):
        for key, col in self._columns.items():
            yield key, self._getcol(key)

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
