"""Graph bolt sampling graph module."""


class Graph(object):             
    r"""Graph bolt sampling graph base class."""

    def sample_negative_edge_uniform(self, ids):
        """
        Sample uniformly from negative edges for given node IDs
        """

    def sample_negative_edge_global_uniform(self, ids):
        """
        Sample uniformly from negative edges for all nodes in the graph
        """

    def sample_full_neighbor(self, ids):
        """
        Sample all neighbors of given nodes
        """

    def sample_neighbor_weighted(self, ids):
        """
        Sample neighbors of given nodes, weighted by edge weights
        """

    def random_walk(self, ids, n_layers):
        """
        Perform random walk starting from given nodes and going through n_layers
        """
