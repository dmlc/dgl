"""Temporary adapter to unify DGLGraph and bipartite HeteroGraph for scheduler.
NOTE(minjie): remove once all scheduler codes are migrated to bipartite
"""
from __future__ import absolute_import

from abc import ABC, abstractmethod

class GraphAdapter(ABC):
    @property
    @abstractmethod
    def gidx(self):
        """Get graph index object."""
        pass

    @abstractmethod
    def num_src(self):
        """Number of source nodes."""
        pass

    @abstractmethod
    def num_dst(self):
        """Number of destination nodes."""
        pass

    @abstractmethod
    def num_edges(self):
        """Number of edges."""
        pass

    @property
    @abstractmethod
    def srcframe(self):
        """Frame to store source node features."""
        pass

    @property
    @abstractmethod
    def dstframe(self):
        """Frame to store source node features."""
        pass

    @property
    @abstractmethod
    def edgeframe(self):
        """Frame to store edge features."""
        pass

    @property
    @abstractmethod
    def msgframe(self):
        """Frame to store messages."""
        pass

    @property
    @abstractmethod
    def msgindicator(self):
        """Message indicator tensor."""
        pass

    @msgindicator.setter
    @abstractmethod
    def msgindicator(self, val):
        """Set new message indicator tensor."""
        pass

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
        pass

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
        pass

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
        pass
