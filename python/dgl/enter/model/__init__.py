
from .gcn import GCN
from .gat import GAT
from .sage import GraphSAGE
from ..utils.factory import ModelFactory

ModelFactory.register("gcn", filename = "gcn.py")(GCN)
ModelFactory.register("gat", filename = "gat.py")(GAT)
ModelFactory.register("sage", filename = "sage.py")(GraphSAGE)
