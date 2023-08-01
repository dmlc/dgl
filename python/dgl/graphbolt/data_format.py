"""Data format enums for graphbolt."""

from enum import Enum

__all__ = ["LinkPredictionEdgeFormat"]


class LinkPredictionEdgeFormat(Enum):
    """
    An Enum class representing the formats of positive and negative edges used
    in link prediction:

    Attributes:
    CONDITIONED: Represents the 'conditioned' format where data is
    structured as quadruples `[u, v, [negative heads], [negative tails]]`
    indicating the source and destination nodes of positive and negative edges.

    INDEPENDENT: Represents the 'independent' format where data is structured
    as triples `[u, v, label]` indicating the source and destination nodes of
    an edge, with a label (0 or 1) denoting it as negative or positive.
    """

    CONDITIONED = "conditioned"
    INDEPENDENT = "independent"
