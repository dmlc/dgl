"""Graph Replicator."""
from torch.fx import Graph, Node
from torch.fx.node import map_arg 


class GraphReplicator(Graph):
    """The Graph Replicator class.

    The graph replicator helps to replicate a torch.fx.Graph object.
    """
    def __init__(self):
        super().__init__()
        self.env = {}

    def arg_transform(self, node: Node):
        """Arg transform function."""
        return self.env[node.name]

    def insert_node_copy(self, node: Node):
        """Insert a node copy to the graph."""
        new_args = map_arg(node.args, self.arg_transform)
        new_node = self.create_node(node.op, node.target, new_args, node.kwargs, node.name)
        self.env[node.name] = new_node
        return new_node

    def insert_input(self, name):
        """Insert an input to the graph."""
        new_node = self.placeholder(name)
        self.env[name] = new_node

    def insert_inputs(self, names):
        """Insert inputs to the graph."""
        for name in names:
            self.insert_input(name)

    def insert_outputs(self, nodes):
        """Insert outputs to the graph."""
        nodes = [self.env[node.name] for node in nodes]
        self.output(nodes[0] if len(nodes) == 1 else tuple(nodes))
