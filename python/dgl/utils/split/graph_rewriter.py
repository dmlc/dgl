from torch.fx import Graph

from .constants import OUTPUT, DGL_GRAPH


class GraphRewriter():
    @staticmethod
    def blocks_to_graph(graph: Graph):
        blocks = None
        for node in graph.nodes:
            if node.node_type == DGL_GRAPH and blocks is None:
                blocks = node
            elif node.node_type == DGL_GRAPH:
                node.replace_all_uses_with(blocks)
                graph.erase_node(node)
        graph.lint()

    @staticmethod
    def remove_unused_nodes(graph: Graph):
        for node in graph.nodes.__reversed__():
            if node.op != OUTPUT:
                if len(node.users) == 0:
                    graph.erase_node(node)
        graph.lint()
