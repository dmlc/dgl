from collections import Mapping
from dgl.storages import wrap_storage
from dgl.utils import recursive_apply

# A GraphStorage class where ndata and edata can be any FeatureStorage but
# otherwise the same as the wrapped DGLGraph.
class OtherFeatureGraphStorage(object):
    def __init__(self, g, ndata=None, edata=None):
        self.g = g
        self._ndata = recursive_apply(ndata, wrap_storage) if ndata is not None else {}
        self._edata = recursive_apply(edata, wrap_storage) if edata is not None else {}

        for k, v in self._ndata.items():
            if not isinstance(v, Mapping):
                assert len(self.g.ntypes) == 1
                self._ndata[k] = {self.g.ntypes[0]: v}
        for k, v in self._edata.items():
            if not isinstance(v, Mapping):
                assert len(self.g.canonical_etypes) == 1
                self._edata[k] = {self.g.canonical_etypes[0]: v}

    def get_node_storage(self, key, ntype=None):
        if ntype is None:
            ntype = self.g.ntypes[0]
        return self._ndata[key][ntype]

    def get_edge_storage(self, key, etype=None):
        if etype is None:
            etype = self.g.canonical_etypes[0]
        return self._edata[key][etype]

    def __getattr__(self, key):
        # I wrote it in this way because I'm too lazy to write "def sample_neighbors"
        # or stuff like that.
        if key in ['ntypes', 'etypes', 'canonical_etypes', 'sample_neighbors',
                   'subgraph', 'edge_subgraph', 'find_edges', 'num_nodes']:
            # Delegate to the wrapped DGLGraph instance.
            return getattr(self.g, key)
        else:
            return super().__getattr__(key)
