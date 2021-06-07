.. _api-geometry:

dgl.geometry
=================================

.. automodule:: dgl.geometry

.. _api-geometry-farthest-point-sampler:

Farthest Point Sampler
-----------

Farthest point sampling is a greedy algorithm that samples from a point cloud
data iteratively. It starts from a random single sample of point. In each iteration,
it samples from the rest points that is the farthest from the set of sampled points.

.. autoclass:: farthest_point_sampler

.. _api-geometry-neighbor-matching:

Neighbor Matching
-----------------------------

Neighbor matching is an important module in the Graclus clustering algorithm.

.. autoclass:: neighbor_matching
