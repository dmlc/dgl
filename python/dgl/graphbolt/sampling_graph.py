"""Sampling Graphs."""

from typing import Dict, Union


class SamplingGraph:
    r"""Class for sampling graph."""

    def __init__(self):
        pass

    @property
    def num_nodes(self) -> Union[int, Dict[str, int]]:
        """The number of nodes in the graph.
        - If the graph is homogenous, returns an integer.
        - If the graph is heterogenous, returns a dictionary.

        Returns
        -------
        Union[int, Dict[str, int]]
            The number of nodes. Integer indicates the total nodes number of a
            homogenous graph; dict indicates nodes number per node types of a
            heterogenous graph.
        """
        raise NotImplementedError

    @property
    def num_edges(self) -> Union[int, Dict[str, int]]:
        """The number of edges in the graph.
        - If the graph is homogenous, returns an integer.
        - If the graph is heterogenous, returns a dictionary.

        Returns
        -------
        Union[int, Dict[str, int]]
            The number of edges. Integer indicates the total edges number of a
            homogenous graph; dict indicates edges number per edge types of a
            heterogenous graph.
        """
        raise NotImplementedError
