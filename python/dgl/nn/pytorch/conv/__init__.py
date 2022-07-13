"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .agnnconv import AGNNConv
from .appnpconv import APPNPConv
from .chebconv import ChebConv
from .edgeconv import EdgeConv
from .gatconv import GATConv
from .gatv2conv import GATv2Conv
from .egatconv import EGATConv
from .ginconv import GINConv
from .gineconv import GINEConv
from .gmmconv import GMMConv
from .graphconv import GraphConv, EdgeWeightNorm
from .nnconv import NNConv
from .relgraphconv import RelGraphConv
from .sageconv import SAGEConv
from .sgconv import SGConv
from .tagconv import TAGConv
from .gatedgraphconv import GatedGraphConv
from .densechebconv import DenseChebConv
from .densegraphconv import DenseGraphConv
from .densesageconv import DenseSAGEConv
from .atomicconv import AtomicConv
from .cfconv import CFConv
from .dotgatconv import DotGatConv
from .twirlsconv import TWIRLSConv, TWIRLSUnfoldingAndAttention
from .gcn2conv import GCN2Conv
from .hgtconv import HGTConv
from .grouprevres import GroupRevRes
from .egnnconv import EGNNConv
from .pnaconv import PNAConv
from .dgnconv import DGNConv

__all__ = ['GraphConv', 'EdgeWeightNorm', 'GATConv', 'GATv2Conv', 'EGATConv', 'TAGConv',
           'RelGraphConv', 'SAGEConv', 'SGConv', 'APPNPConv', 'GINConv', 'GINEConv',
           'GatedGraphConv', 'GMMConv', 'ChebConv', 'AGNNConv', 'NNConv', 'DenseGraphConv',
           'DenseSAGEConv', 'DenseChebConv', 'EdgeConv', 'AtomicConv', 'CFConv', 'DotGatConv',
           'TWIRLSConv', 'TWIRLSUnfoldingAndAttention', 'GCN2Conv', 'HGTConv', 'GroupRevRes',
           'EGNNConv', 'PNAConv', 'DGNConv']
