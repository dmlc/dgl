"""MXNet modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .agnnconv import AGNNConv
from .appnpconv import APPNPConv
from .chebconv import ChebConv
from .densechebconv import DenseChebConv
from .densegraphconv import DenseGraphConv
from .densesageconv import DenseSAGEConv
from .edgeconv import EdgeConv
from .gatconv import GATConv
from .gatedgraphconv import GatedGraphConv
from .ginconv import GINConv
from .gmmconv import GMMConv
from .graphconv import GraphConv
from .nnconv import NNConv
from .relgraphconv import RelGraphConv
from .sageconv import SAGEConv
from .sgconv import SGConv
from .tagconv import TAGConv

__all__ = [
    "GraphConv",
    "TAGConv",
    "RelGraphConv",
    "GATConv",
    "SAGEConv",
    "GatedGraphConv",
    "ChebConv",
    "AGNNConv",
    "APPNPConv",
    "DenseGraphConv",
    "DenseSAGEConv",
    "DenseChebConv",
    "EdgeConv",
    "GINConv",
    "GMMConv",
    "NNConv",
    "SGConv",
]
