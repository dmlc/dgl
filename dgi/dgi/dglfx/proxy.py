import operator
import builtins

from torch.fx import Proxy, Node, Tracer
from ..constants import CALL_METHOD, CALL_FUNCTION, \
    DGL_GRAPH, DGL_GRAPH_ATTRIBUTE


class DGLGraphProxy(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = DGL_GRAPH

    @property
    def is_block(self):
        return True

    def local_var(self):
        return self

    def local_scope(self):
        return self

    def __getitem__(self, rhs):
        return self.tracer.create_proxy(CALL_FUNCTION, operator.getitem, (self, rhs), {}, 
            proxy_factory_fn=self.tracer.dgl_graph_proxy)

    def apply_edges(self, *args, **kwargs):
        return self.tracer.create_proxy(CALL_METHOD, "apply_edges", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_void_call)

    def update_all(self, *args, **kwargs):
        return self.tracer.create_proxy(CALL_METHOD, "update_all", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_void_call)

    @property
    def srcdata(self):
        return self.tracer.create_proxy(CALL_FUNCTION, builtins.getattr, (self, "srcdata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def dstdata(self):
        return self.tracer.create_proxy(CALL_FUNCTION, builtins.getattr, (self, "dstdata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def ndata(self):
        return self.tracer.create_proxy(CALL_FUNCTION, builtins.getattr, (self, "ndata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    @property
    def edata(self):
        return self.tracer.create_proxy(CALL_FUNCTION, builtins.getattr, (self, "edata"), {},
            proxy_factory_fn=self.tracer.dgl_graph_attribute)

    def __str__(self):
        return "{}{}".format(DGL_GRAPH, super().__str__())


class DGLGraphAttribute(Proxy):
    def __init__(self, node: Node, tracer: Tracer = None):
        super().__init__(node, tracer)
        node.node_type = DGL_GRAPH_ATTRIBUTE

    def update(self, *args, **kwargs):
        return self.tracer.create_proxy(CALL_METHOD, "update", (self,) + args, kwargs, 
            proxy_factory_fn=self.tracer.dgl_void_call)

    def __getitem__(self, rhs):
        return self.tracer.create_proxy(CALL_FUNCTION, operator.getitem, (self, rhs), {}, 
            proxy_factory_fn=self.tracer.get_from_dgl_attr)

    def pop(self, rhs):
        return self.tracer.create_proxy(CALL_FUNCTION, operator.getitem, (self, rhs), {}, 
            proxy_factory_fn=self.tracer.get_from_dgl_attr)

    def __str__(self):
        return "{}{}".format(DGL_GRAPH_ATTRIBUTE, super().__str__())
