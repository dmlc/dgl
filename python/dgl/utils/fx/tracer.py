"""DGL Tracer."""
# pylint: disable=no-member
import math

import torch
import dgl.nn
from torch.fx import Tracer, Proxy, Node, GraphModule
from torch.fx._compatibility import compatibility

from .proxy import DGLGraphProxy
from .constants import NORMAL_DATA


class DGLTracer(Tracer):
    """The DGL Tracer Class. Extended from torch.fx.tracer.

    The DGL Tracer can trace a nn.module forward function to a computation graph.
    Arguments are the same as `torch.fx.tracer`.

    Parameters
    ----------
    autowrap_modules : Tuple[ModuleType]
        Defaults to `(math, )`, Python modules whose functions should be wrapped automatically
        without needing to use fx.wrap(). Backward-compatibility for this parameter is guaranteed.
    autowrap_function : Tuple[Callable, ...]
        Python functions that should be wrapped automatically without needing to use fx.wrap().
        Backward compabilibility for this parameter is guaranteed.
    param_shapes_constant : bool
        When this flag is set, calls to shape, size and a few other shape like attributes of a
        module's parameter will be evaluted directly, rather than returning a new Proxy value for
        an attribute access. Backward compatibility for this parameter is guaranteed.
    """
    @compatibility(is_backward_compatible=True)
    def __init__(self, autowrap_modules = (math, ),
                 autowrap_functions = (),
                 param_shapes_constant = False) -> None:
        self.graph_proxy = None
        self.conv_modules = dgl.nn.conv.__dict__["__all__"]
        super().__init__(autowrap_modules, autowrap_functions, param_shapes_constant)

    def set_conv_modules(self, modules):
        """Set Conv modules."""
        if isinstance(modules, (list, tuple)):
            for module in modules:
                self.set_conv_module(module)
        else:
            self.set_conv_module(modules)

    def set_conv_module(self, module):
        """Set Conv module."""
        if not isinstance(module, torch.nn.module):
            raise Exception("Conv Modules must be torch.nn.module.")
        self.conv_modules.append(module.__class__.__name__)

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None) -> Node:
        """Create a node. We tag the node type here."""
        node = super().create_node(kind, target, args, kwargs, name, type_expr)
        node.node_type = NORMAL_DATA
        return node

    def create_proxy(self, kind, target, args, kwargs,
                     name=None, type_expr=None, proxy_factory_fn=None):
        """Create a proxy. We modify the `proxy_factory_fn` as our function."""
        if proxy_factory_fn is None:
            proxy_factory_fn = self.proxy_factory_fn
        return super().create_proxy(kind, target, args, kwargs, name, type_expr, proxy_factory_fn)

    @compatibility(is_backward_compatible=True)
    def proxy_factory_fn(self, node: Node) -> "Proxy":
        """Our default proxy factory fn. We set the first proxy as graph proxy."""
        if self.graph_proxy is None:
            self.graph_proxy = self.dgl_graph_proxy(node)
            return self.graph_proxy
        return Proxy(node, self)

    @compatibility(is_backward_compatible=True)
    def dgl_graph_proxy(self, node: Node) -> "Proxy":
        """Create a DGL graph proxy."""
        return DGLGraphProxy(node, self)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str):
        """If the module is in `self.conv_modules`, we do not enter it."""
        if m.__class__.__name__ in self.conv_modules:
            return True
        return super().is_leaf_module(m, module_qualified_name)


@compatibility(is_backward_compatible=True)
def dgl_symbolic_trace(root, conv_modules = (), concrete_args=None):
    """DGL symbolic trace function.

    We use this function to trace the nn.module to a computation graph. The output is
    a `torch.fx.GraphModule` object.

    Parameters
    ----------
    root : nn.Module
        Module or function to be traced and converted into a Graph representation.
    conv_modules : tuple
        The conv modules that we do not enter.
    concrete_args : Optional[Dict[str, any]]
        Inputs to be partially specialized.

    Returns
    -------
    GraphModule
        a Module created from the recorded operations from ``root``.
    """
    tracer = DGLTracer()
    tracer.set_conv_modules(conv_modules)
    graph = tracer.trace(root, concrete_args)
    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    graph_module = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(graph_module, key):
            setattr(graph_module, key, getattr(root, key))
    return graph_module
