"""Function Generator."""
import types

import dgl  # for holding the environment.
import torch
import torch
from torch.fx import GraphModule, Graph

from ..fx import dgl_symbolic_trace
from .schema import Schema
from .graph_rewriter import GraphRewriter
from .graph_rearranger import GraphRearranger
from .constants import CONV_BLOCK


class FunctionGenerator(torch.nn.Module):
    """The function generator class.

    Can split the forward function to layer-wise sub-functions.
    """
    def __init__(self, debug=False):
        super().__init__()
        self.debug = debug
        self.schema = None
        self.funcs = []

    def module_split(self, module: torch.nn.Module):
        """The module split function.

        Split the forward function of the input module.
        """
        for name in module.__dict__:
            if hasattr(module, name):
                attr = getattr(module, name)
                setattr(self, name, attr)

        if isinstance(module, GraphModule):
            self.traced = module
        else:
            self.traced = dgl_symbolic_trace(module)

        if self.debug:
            print("-------- Origin forward function -------")
            print(self.traced.code.strip())
            print("----------------------------------------")

        self.schema = Schema()
        self.schema.record_inputs_and_outputs(self.traced.graph)
        GraphRewriter.blocks_to_graph(self.traced.graph)
        GraphRewriter.remove_unused_nodes(self.traced.graph)
        self.traced.recompile()

        if self.debug:
            print("------- Modified forward function ------")
            print(self.traced.code.strip())
            print("----------------------------------------")

        rearranger = GraphRearranger(self.traced)
        rearranger.rearrange()
        graphs_list = rearranger.get_splitted_graphs()

        for layer_id, graph in enumerate(graphs_list):
            self.register_func_from_graph(graph, layer_id)
            self.schema.create_layer(graph)

    def register_func_from_graph(self, graph: Graph, layer_id: int):
        """Register function from the computation graph."""
        graph_src = graph.python_code("self").src

        func_name = CONV_BLOCK + str(layer_id)
        graph_src = graph_src.replace("def forward(", "def {}(".format(func_name))
        self.set_function_from_string(graph_src, func_name)

        if self.debug:
            print("--------- Layer {} conv function --------".format(layer_id))
            print(graph_src.strip())
            print("----------------------------------------")

    def set_function_from_string(self, func_src, func_name):
        """Set the function from string."""
        globals_vals = globals()
        exec(func_src, globals_vals)
        setattr(self, func_name, types.MethodType(globals_vals[func_name], self))
        self.funcs.append(getattr(self, func_name))

    def get_schema(self):
        """Get the schema."""
        return self.schema

    def get_funcs(self):
        """Get the splitted functions."""
        return self.funcs
