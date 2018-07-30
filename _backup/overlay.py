
class NodeDictOverlay(MutableMapping):
    def __init__(self, frame):
        self._frame = frame

    @property
    def num_nodes(self):
        return self._frame.num_rows()

    def add_nodes(self, nodes, attrs):
        # NOTE: currently `nodes` are not used. Users need to make sure
        # the node ids are continuous ids from 0.
        # NOTE: this is a good place to hook any graph mutation logic.
        self._frame.append(attrs)

    def delete_nodes(self, nodes):
        # NOTE: this is a good place to hook any graph mutation logic.
        raise NotImplementedError('Delete nodes in the graph is currently not supported.')

    def get_node_attrs(self, nodes, key):
        if nodes == ALL:
            # get the whole column
            return self._frame[key]
        else:
            # TODO(minjie): should not rely on tensor's __getitem__ syntax.
            return utils.id_type_dispatch(
                    nodes,
                    lambda nid : self._frame[key][nid],
                    lambda id_array : self._frame[key][id_array])

    def set_node_attrs(self, nodes, key, val):
        if nodes == ALL:
            # replace the whole column
            self._frame[key] = val
        else:
            # TODO(minjie): should not rely on tensor's __setitem__ syntax.
            utils.id_type_dispatch(
                    nodes,
                    lambda nid : self._frame[key][nid] = val,
                    lambda id_array : self._frame[key][id_array] = val)

    def __getitem__(self, nodes):
        def _check_one(nid):
            if nid >= self.num_nodes:
                raise KeyError
        def _check_many(id_array):
            if F.max(id_array) >= self.num_nodes:
                raise KeyError
        utils.id_type_dispatch(nodes, _check_one, _check_many)
        return utils.MutableLazyDict(
                lambda key: self.get_node_attrs(nodes, key),
                lambda key, val: self.set_node_attrs(nodes, key, val)
                self._frame.schemes)

    def __setitem__(self, nodes, attrs):
        # Happens when adding new nodes in the graph.
        self.add_nodes(nodes, attrs)

    def __delitem__(self, nodes):
        # Happens when deleting nodes in the graph.
        self.delete_nodes(nodes)

    def __len__(self):
        return self.num_nodes

    def __iter__(self):
        raise NotImplementedError()

class AdjOuterOverlay(MutableMapping):
    """
    TODO: Replace this with a more efficient dict structure.
    TODO: Batch graph mutation is not supported.
    """
    def __init__(self):
        self._adj = {}

    def __setitem__(self, u, inner_dict):
        self._adj[u] = inner_dict

    def __getitem__(self, u):
        def _check_one(nid):
            if nid not in self._adj:
                raise KeyError
        def _check_many(id_array):
            pass
        utils.id_type_dispatch(u, _check_one, _check_many)
        return utils.id_type_dispatch(u)

    def __delitem__(self, u):
        # The delitem is ignored.
        raise NotImplementedError('Delete edges in the graph is currently not supported.')

class AdjInnerOverlay(dict):
    """TODO: replace this with a more efficient dict structure."""
    def __setitem__(self, v, attrs):
        pass
