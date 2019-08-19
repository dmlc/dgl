"""This file defines high level nn.model interface provided by DGL.

It is recommended the frameworks implement all the interfaces. However, it is
also OK to skip some. The generated backend module has an ``is_enabled`` function
that returns whether the interface is supported by the framework or not.
"""

###############################################################################
# Softmax module

def edge_softmax(graph, logits):
    """Compute edge softmax"""
    pass