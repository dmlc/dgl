
from .gcn import GCN
from .gat import GAT
from .sage import GraphSAGE
from .sgc import SGC
from .gin import GIN
from ...utils.factory import NodeModelFactory

NodeModelFactory.register("gcn")(GCN)
NodeModelFactory.register("gat")(GAT)
NodeModelFactory.register("sage")(GraphSAGE)
NodeModelFactory.register("sgc")(SGC)
NodeModelFactory.register("gin")(GIN)
