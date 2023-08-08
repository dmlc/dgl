"""Data format enums for graphbolt."""

from enum import Enum

__all__ = ["LinkPredictionEdgeFormat"]


class LinkPredictionEdgeFormat(Enum):
    """
    An Enum class representing the formats of positive and negative edges used
    in link prediction:

    Attributes:
    INDEPENDENT: Represents the 'independent' format where data is structured
    as triples `(u, v, label)` indicating the source and destination nodes of
    an edge, with a label (0 or 1) denoting it as negative or positive.

    CONDITIONED: Represents the 'conditioned' format where data is structured
    as quadruples `(u, v, neg_u, neg_v)` indicating the source and destination
    nodes of positive and negative edges. And 'u' with 'v' are 1D tensors with
    the same shape, while 'neg_u' and 'neg_v' are 2D tensors with the same
    shape.

    HEAD_CONDITIONED: Represents the 'head conditioned' format where data is
    structured as triples `(u, v, neg_u)`, where '(u, v)' signifies the
    source and destination nodes of positive edges, while each node in
    'neg_u' collaborates with 'v' to create negative edges. And 'u' and 'v' are
    1D tensors with the same shape, while 'neg_u' is a 2D tensor.

    TAIL_CONDITIONED: Represents the 'tail conditioned' format where data is
    structured as triples `(u, v, neg_v)`, where '(u, v)' signifies the
    source and destination nodes of positive edges, while 'u' collaborates
    with each node in 'neg_v' to create negative edges. And 'u' and 'v' are
    1D tensors with the same shape, while 'neg_v' is a 2D tensor.
    """

    INDEPENDENT = "independent"
    CONDITIONED = "conditioned"
    HEAD_CONDITIONED = "head_conditioned"
    TAIL_CONDITIONED = "tail_conditioned"
