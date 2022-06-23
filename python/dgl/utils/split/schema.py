from .constants import PLACEHOLDER, OUTPUT


class Schema():
    def __init__(self):
        self.layers = []
        self.name2arg_map = {}
        self.blocks_name = None
        self.first_layer_input = []
        self.last_layer_output = []

    def record_inputs_and_outputs(self, origin_graph):
        for node in origin_graph.nodes:
            if node.op == PLACEHOLDER:
                self.first_layer_input.append(node.name)
            if node.op == OUTPUT:
                args = node.args[0] if isinstance(node.args[0], tuple) else (node.args[0],)
                for node in args:
                    self.last_layer_output.append(node.name)

    def create_layer(self, graph):
        self.layers.append(GraphLayer(self))
        if len(self.layers) != 1:
            self.layers[-2].next_layer = self.curr_layer
        for node in graph.nodes:
            if node.op == PLACEHOLDER:
                self.record_input(node.name)
            elif node.op == OUTPUT:
                args = node.args[0] if isinstance(node.args[0], tuple) else (node.args[0],)
                output_names = [node.name for node in args]
                self.record_outputs(output_names)

    def get_layer(self, id):
        return self.layers[id]

    def record_input(self, name):
        if self.blocks_name is None:
            self.blocks_name = name
        if name not in self.name2arg_map:
            self.name2arg_map[name] = ArgNode(name)
        input_arg = self.name2arg_map[name]
        self.curr_layer.add_input(input_arg)
        input_arg.add_layer(self.curr_layer)

    def record_inputs(self, names):
        for name in names:
            self.record_input(name)

    def record_output(self, name):
        if name in self.name2arg_map:
            raise RuntimeError("The output name is used before!")
        output_arg = ArgNode(name, self.curr_layer)
        self.name2arg_map[name] = output_arg
        self.curr_layer.add_output(output_arg)

    def record_outputs(self, names):
        for name in names:
            self.record_output(name)

    @property
    def curr_layer(self):
        return self.layers[-1]

    @property
    def layers_count(self):
        return len(self.layers)


class GraphLayer():
    def __init__(self, schema: Schema):
        super().__init__()
        self.schema = schema
        self.id = schema.layers_count
        self.inputs: list[ArgNode] = []
        self.outputs: list[ArgNode] = []
        self.next_layer = None

    def next(self):
        return self.next_layer

    def add_input(self, input_arg):
        self.inputs.append(input_arg)

    def add_output(self, output_arg):
        self.outputs.append(output_arg)


class ArgNode():
    # Arg is always create by layer output, or the init
    def __init__(self, name: str, output_layer: GraphLayer = None):
        self.name = name
        self.input_layers = []
        self.output_layer = output_layer

    def add_layer(self, layer: GraphLayer):
        self.input_layers.append(layer)
    
    def __str__(self):
        return "{}, input: {}, output: {}".format(self.name,
                                                  [layer.id for layer in self.input_layers],
                                                  self.output_layer)
