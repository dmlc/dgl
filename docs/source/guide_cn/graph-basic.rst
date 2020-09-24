.. _guide_cn-graph-basic:

1.1 Some Basic Definitions about Graphs (Graphs 101)
1.1 关于图的基本概念
-----------------

A graph :math:`G=(V, E)` is a structure used to represent entities and their relations. It consists of
two sets -- the set of nodes :math:`V` (also called vertices) and the set of edges :math:`E` (also called
arcs). An edge :math:`(u, v) \in E` connecting a pair of nodes :math:`u` and :math:`v` indicates that there is a
relation between them. The relation can either be undirected, e.g., capturing symmetric
relations between nodes, or directed, capturing asymmetric relations. For example, if a
graph is used to model the friendships relations of people in a social network, then the edges
will be undirected as friendship is mutual; however, if the graph is used to model how people
follow each other on Twitter, then the edges are directed. Depending on the edges'
directionality, a graph can be *directed* or *undirected*.

图是用以保存实体及其关系的的结构，记为 :math:`G=(V, E)` 。图由两个集合组成，一是节点的集合 :math:`V` ，一个是边的集合 :math:`E` 。
在边集 :math:`E` 中，一条边 :math:`(u, v) \in E` 连接一对节点 :math:`u` 和 :math:`v` ，表明两节点间存在关系。关系可以是无向的，
如描述节点之间的对称关系；也可以是有向的，如描述非对称关系。例如，若用图对社交网络中人们的友谊关系进行建模，因为友谊是相互的，则边是无向的；
若用图对Twitter用户的关注行为进行建模，则边是有向的。图可以是 *有向的* 或 *无向的* ，这取决于图中边的方向性。

Graphs can be *weighted* or *unweighted*. In a weighted graph, each edge is associated with a
scalar weight. For example, such weights might represent lengths or connectivity strengths.

图可以是 *加权的* 或 *未加权的*。在加权图中，每条边都与一个标量权重值相关联。例如，该权重可以表示长度或连接的强度。

Graphs can also be either *homogeneous* or *heterogeneous*. In a homogeneous graph, all
the nodes represent instances of the same type and all the edges represent relations of the
same type. For instance, a social network is a graph consisting of people and their
connections, representing the same entity type.

图可以是 *同构的* 或是 *异构的*。在同构图中，所有节点表示同一类型的实体，所有边表示同一类型的关系。
例如，社交网络的图由表示同一实体类型的人及其相互之间的连接组成。

In contrast, in a heterogeneous graph, the nodes and edges can be of different types. For
instance, the graph encoding a marketplace will have buyer, seller, and product nodes that
are connected via wants-to-buy, has-bought, is-customer-of, and is-selling edges. The
bipartite graph is a special, commonly-used type of heterogeneous graph, where edges
exist between nodes of two different types. For example, in a recommender system, one can
use a bipartite graph to represent the interactions between users and items. For working
with heterogeneous graphs in DGL, see :ref:`guide-graph-heterogeneous`.

相对地，在异构图中，节点和边的类型可以是不同的。例如，编码市场的图可以有表示"顾客"、"商家"和"商品"的节点，
它们通过“想购买”、“已经购买”、“是的顾客”和“正在销售”的边互相连接。二分图是一类特殊的、常用的异构图，
其中的边连接两类不同类型的节点。例如，在推荐系统中，可以使用二分图表示"用户"和"物品"之间的关系。读者可参考 :ref:`guide_cn-graph-heterogeneous`。

Multigraphs are graphs that can have multiple (directed) edges between the same pair of nodes,
including self loops. For instance, two authors can coauthor a paper in different years,
resulting in edges with different features.

在多重图中，同一对节点之间可以有多条（有向）边，包括自循环的边。例如，两名作者可以在不同年份共同署名文章，
这就带来了具有不同特征的多条边。
