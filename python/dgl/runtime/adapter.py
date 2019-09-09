"""Temporary adapter to unify DGLGraph and HeteroGraph for scheduler.
NOTE(minjie): remove once all scheduler codes are migrated to heterograph
"""
from __future__ import absolute_import

from abc import ABC, abstractmethod

class GraphAdapter(ABC):
    """Temporary adapter class to unify DGLGraph and DGLHeteroGraph for schedulers."""
    @property
    @abstractmethod
    def gidx(self):
        """Get graph index object."""

    @abstractmethod
    def num_src(self):
        """Number of source nodes."""

    @abstractmethod
    def num_dst(self):
        """Number of destination nodes."""

    @abstractmethod
    def num_edges(self):
        """Number of edges."""

    @property
    @abstractmethod
    def srcframe(self):
        """Frame to store source node features."""

    @property
    @abstractmethod
    def dstframe(self):
        """Frame to store source node features."""

    @property
    @abstractmethod
    def edgeframe(self):
        """Frame to store edge features."""

    @property
    @abstractmethod
    def msgframe(self):
        """Frame to store messages."""


    @property
    @abstractmethod
    def msgindicator(self):
        """Message indicator tensor."""

    @msgindicator.setter
    @abstractmethod
    def msgindicator(self, val):
        """Set new message indicator tensor."""

    @abstractmethod
    def in_edges(self, nodes):
        """Get in edges

        Parameters
        ----------
        nodes : utils.Index
            Nodes

        Returns
        -------
        tuple of utils.Index
            (src, dst, eid)
        """

    @abstractmethod
    def out_edges(self, nodes):
        """Get out edges

        Parameters
        ----------
        nodes : utils.Index
            Nodes

        Returns
        -------
        tuple of utils.Index
            (src, dst, eid)
        """

    @abstractmethod
    def edges(self, form):
        """Get all edges

        Parameters
        ----------
        form : str
            "eid", "uv", etc.

        Returns
        -------
        tuple of utils.Index
            (src, dst, eid)
        """

    @abstractmethod
    def get_immutable_gidx(self, ctx):
        """Get immutable graph index for kernel computation.

        Parameters
        ----------
        ctx : DGLContext
            The context of the returned graph.

        Returns
        -------
        GraphIndex

        """

    @abstractmethod
    def bits_needed(self):
        """Return the number of integer bits needed to represent the graph

        Returns
        -------
        int
            The number of bits needed
        """
