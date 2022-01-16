import dgl
from .graph import DGLGraphStorage

# A GraphStorage class where ndata and edata can be any FeatureStorage but
# otherwise the same as DGLGraph.
class OtherFeatureGraphStorage(DGLGraphStorage):
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
