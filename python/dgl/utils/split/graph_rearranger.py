"""Graph Rearranger."""
# pylint: disable=comparison-with-callable
import operator
from torch.fx import GraphModule

from .node_relation import GGraph
from .graph_replicator import GraphReplicator
from .constants import DGL_GRAPH, NORMAL_DATA


class GraphRearranger():
    """The Graph Rearranger class.

    The graph rearranger can split the computation graph.
    """
    def __init__(self, traced: GraphModule):
        self.traced = traced
        self.ggraph = GGraph(traced.graph.nodes)
        self.graphs_list = []

    def inputs_binding(self, nodes):
        """The inputs binding rule."""
        passing_edges = []
        for node in nodes:
            if node.node_type == NORMAL_DATA:
                for out_e in node.out_edges:
                    if node.is_message and out_e.dst.message_degree != node.message_degree:
                        passing_edges.append(out_e)

        for start_e in passing_edges:
            e = start_e
            message_layer = e.src.message_degree
            while True:
                if (e.dst.node.target != operator.getitem and len(e.src.out_edges) != 1) \
                    or e.dst.is_message or not e.allow_break:
                    break
                if len(e.dst.in_edges) != 1:
                    is_same_source = True
                    for in_e in e.dst.in_edges:
                        if in_e.src.message_degree != e.src.message_degree:
                            is_same_source = False
                    if not is_same_source:
                        break
                e.dst.message_degree = message_layer
                e = e.dst.out_edges[0]

    def generate_new_graphs(self, nodes):
        """Generate new sub-graphs according too message degrees."""
        message_layers = [[] for _ in range(self.ggraph.max_message + 1)]
        layers_inputs = [set() for _ in range(self.ggraph.max_message + 1)]
        layers_outputs = [set() for _ in range(self.ggraph.max_message + 1)]
        for node in nodes:
            message_layers[node.message_degree].append(node)
            for e in node.out_edges:
                if node.message_degree != e.dst.message_degree:
                    layers_inputs[e.dst.message_degree].add(node)
                    if node.node_type != DGL_GRAPH:
                        layers_outputs[node.message_degree].add(node)

        for i, (inputs, layer_nodes, outputs) in \
            enumerate(zip(layers_inputs, message_layers, layers_outputs)):
            curr_graph = GraphReplicator()
            for input_node in inputs:
                curr_graph.insert_input(input_node.name)
            for node in layer_nodes:
                curr_graph.insert_node_copy(node.node)
            if i != self.ggraph.max_message:
                curr_graph.insert_outputs(outputs)
            curr_graph.lint()
            self.graphs_list.append(curr_graph)

    def get_splitted_graphs(self):
        """Get the splitted graphs."""
        return self.graphs_list

    def rearrange(self):
        """The rearrange function."""
        self.inputs_binding(self.ggraph.nodes)
        self.generate_new_graphs(self.ggraph.nodes)
