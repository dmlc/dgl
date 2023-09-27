"""Sampling Graphs."""

from typing import Dict, Union


class SamplingGraph:
    r"""Class for sampling graph."""

    def __init__(self):
        pass

    @property
    def num_nodes(self) -> Union[int, Dict[str, int]]:
        """
        >   For a homogenous graph, returns the number of nodes in it.
        >   For a Heterogenous graph, returns a dictionary indicating
            the numbers of all types of nodes in it.

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
        """
        >   For a homogenous graph, returns the number of edges in it.
        >   For a Heterogenous graph, returns a dictionary indicating
            the numbers of all types of edges in it.

        Returns
        -------
        Union[int, Dict[str, int]]
            The number of edges in the entire graph (homogenous), or
            a dictionary indicating the numbers of all types of edges
            in the graph (heterogenous).
        """
        raise NotImplementedError
