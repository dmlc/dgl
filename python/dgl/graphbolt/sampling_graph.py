"""Sampling Graphs."""

from typing import Dict, Union

import torch


__all__ = ["SamplingGraph"]


class SamplingGraph:
    r"""Class for sampling graph."""

    def __init__(self):
        pass

    def __repr__(self) -> str:
        """Return a string representation of the graph.

        Returns
        -------
        str
            String representation of the graph.
        """
        raise NotImplementedError

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

    def copy_to_shared_memory(self, shared_memory_name: str) -> "SamplingGraph":
        """Copy the graph to shared memory.

        Parameters
        ----------
        shared_memory_name : str
            Name of the shared memory.

        Returns
        -------
        SamplingGraph
            The copied SamplingGraph object on shared memory.
        """
        raise NotImplementedError

    # pylint: disable=invalid-name
    def to(self, device: torch.device) -> "SamplingGraph":
        """Copy graph to the specified device.

        Parameters
        ----------
        device : torch.device
            The destination device.

        Returns
        -------
        SamplingGraph
            The graph on the specified device.
        """
        raise NotImplementedError
