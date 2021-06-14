.. _guide-graph-basic:

1.1 Some Basic Definitions about Graphs (Graphs 101)
----------------------------------------------------

:ref:`(中文版)<guide_cn-graph-basic>`

A graph :math:`G=(V, E)` is a structure used to represent entities and their relations. It consists of
two sets -- the set of nodes :math:`V` (also called vertices) and the set of edges :math:`E` (also called
arcs). An edge :math:`(u, v) \in E` connecting a pair of nodes :math:`u` and :math:`v` indicates that there is a
relation between them. The relation can either be undirected, e.g., capturing symmetric
relations between nodes, or directed, capturing asymmetric relations. For example, if a
graph is used to model the friendships relations of people in a social network, then the edges
will be undirected as friendship is mutual; however, if the graph is used to model how people
follow each other on Twitter, then the edges are directed. Depending on the edges'
directionality, a graph can be *directed* or *undirected*.

Graphs can be *weighted* or *unweighted*. In a weighted graph, each edge is associated with a
scalar weight. For example, such weights might represent lengths or connectivity strengths.

Graphs can also be either *homogeneous* or *heterogeneous*. In a homogeneous graph, all
the nodes represent instances of the same type and all the edges represent relations of the
same type. For instance, a social network is a graph consisting of people and their
connections, representing the same entity type.

In contrast, in a heterogeneous graph, the nodes and edges can be of different types. For
instance, the graph encoding a marketplace will have buyer, seller, and product nodes that
are connected via wants-to-buy, has-bought, is-customer-of, and is-selling edges. The
bipartite graph is a special, commonly-used type of heterogeneous graph, where edges
exist between nodes of two different types. For example, in a recommender system, one can
use a bipartite graph to represent the interactions between users and items. For working
with heterogeneous graphs in DGL, see :ref:`guide-graph-heterogeneous`.

Multigraphs are graphs that can have multiple (directed) edges between the same pair of nodes,
including self loops. For instance, two authors can coauthor a paper in different years,
resulting in edges with different features.
