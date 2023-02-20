from ...utils.factory import NodeModelFactory
from .gat import GAT
from .gcn import GCN
from .gin import GIN
from .sage import GraphSAGE
from .sgc import SGC

NodeModelFactory.register("gcn")(GCN)
NodeModelFactory.register("gat")(GAT)
NodeModelFactory.register("sage")(GraphSAGE)
NodeModelFactory.register("sgc")(SGC)
NodeModelFactory.register("gin")(GIN)
