import math

import torch
import dgl.nn
from torch.fx import Tracer, GraphModule, Proxy
from torch.fx._compatibility import compatibility


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
                 param_shapes_constant = False):
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
        is_module = False
        for clazz in module.__mro__:
            if clazz == torch.nn.modules.module.Module:
                is_module = True
        if not is_module:
            raise Exception("Conv Modules must be torch.nn.module.")
        self.conv_modules.append(module.__name__)

    @compatibility(is_backward_compatible=True)
    def call_module(self, m: torch.nn.Module, forward, args, kwargs):
        """Call modules."""
        def tag_conv_fn(node):
            node.is_conv = True
            return Proxy(node)

        if m.__class__.__name__ in self.conv_modules:
            module_qualified_name = self.path_of_module(m)
            return self.create_proxy('call_module', module_qualified_name, args, kwargs, \
                proxy_factory_fn=tag_conv_fn)
        super().call_module(m, forward, args, kwargs)


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
    for node in graph.nodes:
        if not hasattr(node, "is_conv"):
            node.is_conv = False

    name = root.__class__.__name__ if isinstance(root, torch.nn.Module) else root.__name__
    graph_module = GraphModule(tracer.root, graph, name)
    for key in dir(root):
        if not hasattr(graph_module, key):
            setattr(graph_module, key, getattr(root, key))
    return graph_module
