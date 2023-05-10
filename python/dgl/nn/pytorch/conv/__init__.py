"""Torch modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .agnnconv import AGNNConv
from .appnpconv import APPNPConv
from .atomicconv import AtomicConv
from .cfconv import CFConv
from .chebconv import ChebConv
from .cugraph_gatconv import CuGraphGATConv
from .cugraph_relgraphconv import CuGraphRelGraphConv
from .cugraph_sageconv import CuGraphSAGEConv
from .densechebconv import DenseChebConv
from .densegraphconv import DenseGraphConv
from .densesageconv import DenseSAGEConv
from .dgnconv import DGNConv
from .dotgatconv import DotGatConv
from .edgeconv import EdgeConv
from .edgegatconv import EdgeGATConv
from .egatconv import EGATConv
from .egnnconv import EGNNConv
from .gatconv import GATConv
from .gatedgcnconv import GatedGCNConv
from .gatedgraphconv import GatedGraphConv
from .gatv2conv import GATv2Conv
from .gcn2conv import GCN2Conv
from .ginconv import GINConv
from .gineconv import GINEConv
from .gmmconv import GMMConv
from .graphconv import EdgeWeightNorm, GraphConv
from .grouprevres import GroupRevRes
from .hgtconv import HGTConv
from .nnconv import NNConv
from .pnaconv import PNAConv
from .relgraphconv import RelGraphConv
from .sageconv import SAGEConv
from .sgconv import SGConv
from .tagconv import TAGConv
from .twirlsconv import TWIRLSConv, TWIRLSUnfoldingAndAttention

__all__ = [
    "GraphConv",
    "EdgeWeightNorm",
    "GATConv",
    "GATv2Conv",
    "EGATConv",
    "EdgeGATConv",
    "TAGConv",
    "RelGraphConv",
    "SAGEConv",
    "SGConv",
    "APPNPConv",
    "GINConv",
    "GINEConv",
    "GatedGraphConv",
    "GatedGCNConv",
    "GMMConv",
    "ChebConv",
    "AGNNConv",
    "NNConv",
    "DenseGraphConv",
    "DenseSAGEConv",
    "DenseChebConv",
    "EdgeConv",
    "AtomicConv",
    "CFConv",
    "DotGatConv",
    "TWIRLSConv",
    "TWIRLSUnfoldingAndAttention",
    "GCN2Conv",
    "HGTConv",
    "GroupRevRes",
    "EGNNConv",
    "PNAConv",
    "DGNConv",
    "CuGraphGATConv",
    "CuGraphRelGraphConv",
    "CuGraphSAGEConv",
]
