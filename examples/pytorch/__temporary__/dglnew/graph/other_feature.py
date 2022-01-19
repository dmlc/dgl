
# A GraphStorage class where ndata and edata can be any FeatureStorage but
# otherwise the same as the wrapped DGLGraph.
class OtherFeatureGraphStorage(object):
    def __init__(self, g, ndata=None, edata=None):
        self.g = g
        self._ndata = ndata or {}
        self._edata = edata or {}

    @property
    def ndata(self):
        return self._ndata

    @property
    def edata(self):
        return self._edata

    def __getattr__(self, key):
        # I wrote it in this way because I'm too lazy to write "def sample_neighbors"
        # or stuff like that.
        if key in ['ntypes', 'etypes', 'canonical_etypes', 'sample_neighbors',
                   'subgraph', 'edge_subgraph', 'find_edges', 'num_nodes']:
            # Delegate to the wrapped DGLGraph instance.
            return getattr(self.g, key)
        else:
            return super().__getattr__(key)
