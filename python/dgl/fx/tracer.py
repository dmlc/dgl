import math

import torch
from torch.fx import Tracer, Proxy, Node, GraphModule

from .proxy import DGLGraphProxy
from .constants import DGL_GRAPH, DGL_GRAPH_DATA, TENSOR_DATA, MASSAGE_PASSING, CALL_METHOD, CALL_MODULE


class DGLTracer(Tracer):
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False) -> None:
        self.graph_proxy = None
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def trace(self, root, concrete_args):
        graph = super().trace(root, concrete_args)
        # static analysis tag nodes
        for node in graph.nodes:
            # tag graph call method. e.g., edge_softmax(g, x); g.number_of_nodes()
            if node.node_type == TENSOR_DATA and node.op == CALL_METHOD and node.args[0].node_type == DGL_GRAPH:
                node.node_type = DGL_GRAPH_DATA
            if node.op == CALL_MODULE and node.args[0].node_type == DGL_GRAPH:
                node.node_type = MASSAGE_PASSING
        return graph

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        node.node_type = TENSOR_DATA
        return node

    def create_proxy(self, kind, target, args, kwargs,
                     name=None, type_expr=None, proxy_factory_fn=None):
        if proxy_factory_fn is None:
            proxy_factory_fn = self.proxy_factory_fn
        return super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

    def proxy_factory_fn(self, node: Node) -> "Proxy":
        if self.graph_proxy is None:
            self.graph_proxy = self.dgl_graph_proxy(node)
            return self.graph_proxy
        return Proxy(node, self)

    def dgl_graph_proxy(self, node: Node) -> "Proxy":
        return DGLGraphProxy(node, self)

    # default tracer
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        return True


def dgl_symbolic_trace(root, concrete_args=None):
    tracer = DGLTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    gm = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(gm, key):
            setattr(gm, key, getattr(root, key))
    return gm
