"""DGL graph proxy."""
import operator
from torch.fx import Proxy, Node, Tracer
from .constants import CALL_FUNCTION, DGL_GRAPH


class DGLGraphProxy(Proxy):
    """The DGLGraph Proxy.

    More functions can be supported here.
    """
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = DGL_GRAPH

    def local_var(self):
        """Local var function."""
        return self

    def local_scope(self):
        """Local scope function."""
        return self

    def __getitem__(self, rhs):
        """Getitem function.
        
        Here we need to specified the output. Since it is also a graph proxy.
        """
        return self.tracer.create_proxy(CALL_FUNCTION, operator.getitem, (self, rhs), {},
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    def apply_edges(self, *args, **kwargs):
        """Apply edges function, not supported."""
        raise RuntimeError("Not support DGLHeteroGraph operation apply_edges!")

    def update_all(self, *args, **kwargs):
        """Update all function, not supported."""
        raise RuntimeError("Not support DGLHeteroGraph operation update_all!")

    @property
    def srcdata(self):
        """Not supported graph attribute srcdata."""
        raise RuntimeError("Not support DGLHeteroGraph attribute srcdata!")

    @property
    def dstdata(self):
        """Not supported graph attribute dstdata."""
        raise RuntimeError("Not support DGLHeteroGraph attribute dstdata!")

    @property
    def ndata(self):
        """Not supported graph attribute ndata."""
        raise RuntimeError("Not support DGLHeteroGraph attribute ndata!")

    @property
    def edata(self):
        """Not supported graph attribute edata."""
        raise RuntimeError("Not support DGLHeteroGraph attribute edata!")

    def __str__(self):
        return "{}{}".format(DGL_GRAPH, super().__str__())
