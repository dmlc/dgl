"""MXNet modules for graph convolutions."""
# pylint: disable= no-member, arguments-differ, invalid-name

from .graphconv import GraphConv
from .relgraphconv import RelGraphConv
from .tagconv import TAGConv

__all__ = ['GraphConv', 'TAGConv', 'RelGraphConv']
