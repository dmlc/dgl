import dgl
import torch
from torch.fx import GraphModule, Interpreter

# TODO: maybe insert to the splitter later.
class CostEvaluater(Interpreter):
    def __init__(self, gm: GraphModule):
        super().__init__(gm)

    def eval(self, *args):
        self.run(*args)
