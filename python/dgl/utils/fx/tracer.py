import math

import torch
import dgl.nn
from torch.fx import Tracer, Proxy, Node, GraphModule
from torch.fx._compatibility import compatibility

from .proxy import DGLGraphProxy
from .constants import DGL_GRAPH, NORMAL_DATA, CALL_FUNCTION, CALL_METHOD, GET_ATTR


class DGLTracer(Tracer):
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False) -> None:
        self.graph_proxy = None
        self.conv_modules = dgl.nn.conv.__dict__["__all__"]
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def set_conv_modules(self, modules):
        if isinstance(modules, tuple) or isinstance(modules, list):
            for module in modules:
                self.set_conv_module(module)
        else:
            self.set_conv_module(modules)

    def set_conv_module(self, module):
        if not isinstance(module, torch.nn.module):
            raise Exception("Conv Modules must be torch.nn.module.")
        self.conv_modules.append(module.__class__.__name__)

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        node.node_type = NORMAL_DATA
        return node

    def create_proxy(self, kind, target, args, kwargs,
                     name=None, type_expr=None, proxy_factory_fn=None):
        if proxy_factory_fn is None:
            proxy_factory_fn = self.proxy_factory_fn
        return super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

    @compatibility(is_backward_compatible=True)
    def proxy_factory_fn(self, node: Node) -> "Proxy":
        if self.graph_proxy is None:
            self.graph_proxy = self.dgl_graph_proxy(node)
            return self.graph_proxy
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def dgl_graph_proxy(self, node: Node) -> "Proxy":
        return DGLGraphProxy(node, self)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        if m.__class__.__name__ in self.conv_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


@compatibility(is_backward_compatible=True)
def dgl_symbolic_trace(root, conv_modules = (), concrete_args=None):
    tracer = DGLTracer()
    tracer.set_conv_modules(conv_modules)
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    gm = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(gm, key):
            setattr(gm, key, getattr(root, key))
    return gm
