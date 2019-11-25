"""MXNet modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .graphconv import GraphConv
from .relgraphconv import RelGraphConv
from .tagconv import TAGConv
from .gatconv import GATConv
from .sageconv import SAGEConv
from .gatedgraphconv import GatedGraphConv
from .chebconv import ChebConv
from .agnnconv import AGNNConv
from .appnpconv import APPNPConv
from .densegraphconv import DenseGraphConv
from .densesageconv import DenseSAGEConv
from .densechebconv import DenseChebConv
from .edgeconv import EdgeConv
from .ginconv import GINConv
from .gmmconv import GMMConv
from .nnconv import NNConv
from .sgconv import SGConv

__all__ = ['GraphConv', 'TAGConv', 'RelGraphConv', 'GATConv',
           'SAGEConv', 'GatedGraphConv', 'ChebConv', 'AGNNConv',
           'APPNPConv', 'DenseGraphConv', 'DenseSAGEConv', 'DenseChebConv',
           'EdgeConv', 'GINConv', 'GMMConv', 'NNConv', 'SGConv']
