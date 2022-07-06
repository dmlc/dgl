"""Node Relation."""
from torch.fx import Node

from .constants import GET_ATTR, CALL_MODULE, OUTPUT, PLACEHOLDER, DGL_GRAPH


def arg_trace(a):
    """Trace the args for a node, return a node set."""
    ret = set()
    if isinstance(a, Node):
        ret.add(a)
    if isinstance(a, dict):
        for _, v in a.items():
            ret = ret.union(arg_trace(v))
    if isinstance(a, tuple) or isinstance(a, list):
        for v in a:
            ret = ret.union(arg_trace(v))
    elif isinstance(a, slice):
        ret = ret.union(arg_trace((a.start, a.step, a.stop)))
    return ret


class GGraph:
    """The class to represent the node relation."""
    def __init__(self, node_list):
        self.nodes = []
        self.name2gnode_map = {}
        self.output = None
        self.inputs = []
        self.get_node_relation(node_list)

    def get_node_relation(self, node_list):
        """Tag nodes and compute their message degree."""
        for lineno, node in enumerate(node_list):
            self.nodes.append(GNode(node, lineno))
            self.name2gnode_map[node.name] = self.nodes[-1]

        for node in self.nodes:
            args = arg_trace(node.args)
            for arg in args:
                allow_break = self.check_allow_break(self.name2gnode_map[arg.name], node)
                self.add_edge(self.name2gnode_map[arg.name], node, allow_break)

        self.tag_nodes()
        self.compute_message_degree()

    def tag_nodes(self):
        """Static tags for nodes."""
        for node in self.nodes:
            if node.op == PLACEHOLDER:
                self.inputs.append(node)
                if node.node_type != DGL_GRAPH:
                    node.message_degree = 0
            # Tag message passing nodes.
            if node.op == CALL_MODULE:
                for e in node.in_edges:
                    if e.src.node_type == DGL_GRAPH:
                        node.is_message = True
            if node.op == OUTPUT:
                self.output = node

    def add_edge(self, src, dst, allow_break=True):
        """Add an edge in the graph."""
        edge = GEdge(src, dst, allow_break)
        src.add_out_edge(edge)
        dst.add_in_edge(edge)

    def check_allow_break(self, src, dst):
        """Define rules that whether the edge can break."""
        # Could add more rules here.
        if src.op == GET_ATTR:
            return False
        return True

    def compute_message_degree(self):
        """Compute message degree for the graph."""
        for node in self.nodes:
            for oe in node.out_edges:
                oe.dst.message_degree = max(oe.dst.message_degree, node.message_degree + oe.src.is_message)
            for ie in node.in_edges:
                if ie.src.message_degree == -1:
                    ie.src.message_degree = node.message_degree

        # remove the last layer
        for node in self.nodes:
            if node.message_degree == self.output.message_degree:
                node.message_degree -= 1

    @property
    def max_message(self):
        """The maximum message degree in the graph."""
        return self.output.message_degree

    def __str__(self):
        ret = ""
        for node in self.nodes:
            ret += "{}; message degree = {}{}\n".format(node.__str__(),
                node.message_degree, " (is message)" if node.is_message else "")
        return ret

class GNode:
    """The class to represent a single node."""
    def __init__(self, node: Node, lineno):
        self.node = node
        self.lineno = lineno                # Lineno in the computation graph.
        self.in_edges = []                  # Input edges list.
        self.out_edges = []                 # Output edges list.
        self.node_type = node.node_type     # Node type.
        self.is_message = False             # Whether it is a message passing node.
        self.message_degree = -1            # The message passing degree.
        # original node attributes.
        self.name = node.name
        self.op = node.op
        self.args = node.args
        self.kwargs = node.kwargs
        self.target = node.target

    def add_in_edge(self, e):
        """Add an input edge for the node."""
        self.in_edges.append(e)

    def add_out_edge(self, e):
        """Add an output edge for the node."""
        self.out_edges.append(e)

    def __str__(self):
        return "{} {} {}: {} {}".format(self.lineno, self.name, self.node_type, 
            [(("" if e.allow_break else "+") + str(e.src.lineno)) for e in self.in_edges],
            [(("" if e.allow_break else "+") + str(e.dst.lineno)) for e in self.out_edges])

    def __repr__(self):
        return self.name

class GEdge:
    """The class to represent an edge in the graph."""
    def __init__(self, src: GNode, dst: GNode, allow_break=True):
        self.src = src
        self.dst = dst
        self.allow_break = allow_break

    def __repr(self):
        return "{} - {}".format(self.src.name, self.dst.name)

    def __str__(self):
        return "{} - {}".format(self.src.name, self.dst.name)
