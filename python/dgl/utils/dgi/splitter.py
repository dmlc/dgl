"""Function Generator."""
# pylint: disable=comparison-with-callable
import operator
from torch.fx import GraphModule, Graph
from torch.fx.passes.split_utils import split_by_tags

from .constants import CONV_BLOCK, CALL_FUNCTION, OUTPUT, PLACEHOLDER


class Splitter():
    """The splitter class.

    Can split the forward function to layer-wise sub-functions.
    """
    def compute_message_degree(self, traced: GraphModule):
        """Compute message degrees."""
        # Set message degree to zero.
        for node in traced.graph.nodes:
            node.message_degree = 0
        for node in traced.graph.nodes:
            for user in node.users:
                user.message_degree = max(user.message_degree, node.message_degree + node.is_conv)
        # Fixed node that do not tagged (e.g., g.number_of_nodes()).
        for node in traced.graph.nodes.__reversed__():
            for arg in node.all_input_nodes:
                if arg.op == PLACEHOLDER or arg.is_conv:
                    continue
                if arg.message_degree != node.message_degree:
                    arg.message_degree = node.message_degree
        # Remove the last layer.
        output_message = max([node.message_degree for node in traced.graph.nodes])
        for node in traced.graph.nodes:
            if node.message_degree == output_message:
                node.message_degree -= 1

    def tag_by_message_degree(self, traced):
        """Tag according to the message degrees."""
        tags = []
        for node in traced.graph.nodes:
            if node.op == PLACEHOLDER or node.op == OUTPUT:
                continue
            node.tag = CONV_BLOCK + str(node.message_degree)
            if node.tag not in tags:
                tags.append(node.tag)
        return tags

    def split(self, traced: GraphModule):
        """The split function."""
        self.compute_message_degree(traced)
        # TODO: Input bindings could be done here.
        tags = self.tag_by_message_degree(traced)
        splitted = split_by_tags(traced, tags)
        return tags, splitted

def blocks_to_graph(graph: Graph):
    """Transform blocks to a graph."""
    graph_list = None
    for node in graph.nodes:
        if node.is_conv:
            graph_obj = node.args[0]
            if graph_obj.op == CALL_FUNCTION and graph_obj.target == operator.getitem:
                graph_list = graph_obj.args[0]
                break
    if graph_list is not None:
        for node in graph.nodes:
            if node.op == CALL_FUNCTION and node.target == operator.getitem \
                and node.args[0] == graph_list:
                node.replace_all_uses_with(graph_list)
                graph.erase_node(node)
        graph.lint()

def split_module(traced: GraphModule, debug=False):
    """The module split function.

    Split the forward function of the input module.

    Parameters
    ----------
    traced : GraphModule
        Module or function to be spiltted.
    debug : bool
        Whether display the debug messages.

    Returns
    ----------
    GraphModule
        The splitted graph module.
    """
    if debug:
        print("-------- Origin forward function -------")
        print(traced.code.strip())
        print("-"*40)

    blocks_to_graph(traced.graph)
    traced.recompile()

    if debug:
        print("------- Modified forward function ------")
        print(traced.code.strip())
        print("-"*40)

    splitter = Splitter()
    tags, splitted = splitter.split(traced)

    if debug:
        print("------------ Main function -------------")
        print(splitted.code.strip())
        print("-"*40)
        for layer_id, tag in enumerate(tags):
            print("--------- Layer {} conv function --------".format(layer_id))
            print(getattr(splitted, tag).code.strip())
            print("-"*40)

    return tags, splitted
