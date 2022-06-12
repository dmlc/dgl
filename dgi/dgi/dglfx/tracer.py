import math

import torch
import dgl.nn
from torch.fx import Tracer, Proxy, Node, GraphModule
from torch.fx._compatibility import compatibility
from dgl.nn.functional import edge_softmax
from dgl.function.message import BinaryMessageFunction, CopyMessageFunction
from dgl.function.reducer import SimpleReduceFunction

from .proxy import DGLGraphProxy, DGLGraphAttribute
from ..constants import DGL_GRAPH, DGL_TENSOR_DATA, DGL_VOID_CALL, DGL_FUNCTION, DGL_GRAPH_DATA, \
    CALL_FUNCTION, CALL_METHOD, GET_ATTR, TENSOR_DATA, UTIL_DATA

def is_dgl_function(target):
    if isinstance(target, SimpleReduceFunction) \
    or isinstance(target, BinaryMessageFunction) \
    or isinstance(target, CopyMessageFunction):
        return True
    return False

def get_dgl_function_kwargs(func):
    if isinstance(func, CopyMessageFunction):
        return {"target": func.target,
                "in_field": func.in_field,
                "out_field": func.out_field}
    if isinstance(func, BinaryMessageFunction):
        return {"binary_op": func.binary_op,
                "lhs": func.lhs,
                "rhs": func.rhs,
                "lhs_field": func.lhs_field,
                "rhs_field": func.rhs_field,
                "out_field": func.out_field}
    if isinstance(func, SimpleReduceFunction):
        return {"name": func._name,
                "msg_field": func.msg_field,
                "out_field": func.out_field}


class DGLTracer(Tracer):
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False) -> None:
        self.graph_proxy = None
        self.conv_modules = dgl.nn.conv.__dict__["__all__"]
        autowrap_functions += (edge_softmax,)
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def trace(self, root, concrete_args):
        graph = super().trace(root, concrete_args)
        # static analysis tag nodes
        for node in graph.nodes:
            # tag graph call method. e.g., edge_softmax(g, x); g.number_of_nodes()
            if node.node_type == TENSOR_DATA and node.op == CALL_METHOD and node.args[0].node_type == DGL_GRAPH:
                node.node_type = DGL_GRAPH_DATA
            # tag x.shape to util_data
            elif node.node_type == TENSOR_DATA and node.op == CALL_FUNCTION \
                and len(node.args) == 2 and node.args[1] == "shape":
                node.node_type = UTIL_DATA
            # tag get_attr to util_data
            elif node.op == GET_ATTR:
                node.node_type = UTIL_DATA
        return graph

    def set_conv_module(self, module):
        self.conv_modules.append(module.__class__.__name__)

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None) -> Node:
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        node.node_type = TENSOR_DATA
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

    @compatibility(is_backward_compatible=True)
    def dgl_graph_attribute(self, node: Node) -> "Proxy":
        return DGLGraphAttribute(node, self)

    @compatibility(is_backward_compatible=True)
    def dgl_void_call(self, node: Node) -> "Proxy":
        node.node_type = DGL_VOID_CALL
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def get_from_dgl_attr(self, node: Node) -> "Proxy":
        node.node_type = DGL_TENSOR_DATA
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def create_arg(self, a):
        if is_dgl_function(a):
            proxy = self.create_proxy(CALL_FUNCTION, a.__class__, (),
                get_dgl_function_kwargs(a), a.name,proxy_factory_fn=self.proxy_factory_fn)
            proxy.node.node_type = DGL_FUNCTION
            return proxy.node
        else:
            return super().create_arg(a)

    # default tracer
    # TODO: current cannot trace dgl functions.
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        return True
        # if m.__class__.__name__ in self.conv_modules:
        #     return True
        # return super().is_leaf_module(m, module_qualified_name)


@compatibility(is_backward_compatible=True)
def dgl_symbolic_trace(root, concrete_args=None):
    tracer = DGLTracer()
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    gm = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(gm, key):
            setattr(gm, key, getattr(root, key))
    return gm
