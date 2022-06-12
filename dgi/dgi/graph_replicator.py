from torch.fx import Graph, Node
from torch.fx.node import map_arg 


class GraphReplicator(Graph):
    def __init__(self):
        super().__init__()
        self.env = {}

    def arg_transform(self, node: Node):
        return self.env[node.name]

    def insert_node_copy(self, node: Node):
        new_args = map_arg(node.args, self.arg_transform)
        new_node = self.create_node(node.op, node.target, new_args, node.kwargs, node.name)
        self.env[node.name] = new_node
        return new_node

    def insert_input(self, name):
        new_node = self.placeholder(name)
        self.env[name] = new_node

    def insert_inputs(self, names):
        for name in names:
            self.insert_input(name)

    def insert_output(self, nodes):
        nodes = [self.env[node.name] for node in nodes]
        self.output(nodes[0] if len(nodes) == 1 else tuple(nodes))
