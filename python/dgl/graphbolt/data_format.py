"""Data format enums for graphbolt."""

from enum import Enum

__all__ = ["LinkPredictionEdgeFormat"]


class LinkPredictionEdgeFormat(Enum):
    """
    An Enum class representing the formats of positive and negative edges used
    in link prediction:

    Attributes:
    INDEPENDENT: Represents the 'independent' format where data is structured
    as triples `[u, v, label]` indicating the source and destination nodes of
    an edge, with a label (0 or 1) denoting it as negative or positive.

    CONDITIONED: Represents the 'conditioned' format where data is
    structured as quadruples `[u, v, [negative heads], [negative tails]]`
    indicating the source and destination nodes of positive and negative edges.

    HEADCONDITIONED: Represents the 'head conditioned' format where data is
    structured as triples `[u, v, [negative heads]]`, where  '(u, v)' signifies
    the source and destination nodes of positive edges, while each node in
    '[negative heads]' collaborates with 'v' to create negative edges.

    TAILCONDITIONED: Represents the 'conditioned' format where data is
    structured as triples `[u, v, [negative tails]]`, where  '(u, v)' signifies
    the source and destination nodes of positive edges, while 'u' collaborates
    with each node in '[negative tails]' to create negative edges.
    """

    INDEPENDENT = "independent"
    CONDITIONED = "conditioned"
    HEADCONDITIONED = "headconditioned"
    TAILCONDITIONED = "tailconditioned"
