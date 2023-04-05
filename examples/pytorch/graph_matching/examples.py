import dgl
import numpy as np
from ged import graph_edit_distance

src1 = [0, 1, 2, 3, 4, 5]
dst1 = [1, 2, 3, 4, 5, 6]

src2 = [0, 1, 3, 4, 5]
dst2 = [1, 2, 4, 5, 6]


G1 = dgl.DGLGraph((src1, dst1))
G2 = dgl.DGLGraph((src2, dst2))


# Exact edit distance with astar search
distance, node_mapping, edge_mapping = graph_edit_distance(
    G1, G1, algorithm="astar"
)
print(distance)  # 0.0
distance, node_mapping, edge_mapping = graph_edit_distance(
    G1, G2, algorithm="astar"
)
print(distance)  # 1.0

# With user-input cost matrices
node_substitution_cost = np.empty((G1.num_nodes(), G2.num_nodes()))
G1_node_deletion_cost = np.empty(G1.num_nodes())
G2_node_insertion_cost = np.empty(G2.num_nodes())

edge_substitution_cost = np.empty((G1.num_edges(), G2.num_edges()))
G1_edge_deletion_cost = np.empty(G1.num_edges())
G2_edge_insertion_cost = np.empty(G2.num_edges())

# Node substitution cost of 0 when node-ids are same, else 1
node_substitution_cost.fill(1.0)
for i in range(G1.num_nodes()):
    for j in range(G2.num_nodes()):
        node_substitution_cost[i, j] = 0.0

# Node insertion/deletion cost of 1
G1_node_deletion_cost.fill(1.0)
G2_node_insertion_cost.fill(1.0)

# Edge substitution cost of 0
edge_substitution_cost.fill(0.0)

# Edge insertion/deletion cost of 0.5
G1_edge_deletion_cost.fill(0.5)
G2_edge_insertion_cost.fill(0.5)

distance, node_mapping, edge_mapping = graph_edit_distance(
    G1,
    G2,
    node_substitution_cost,
    edge_substitution_cost,
    G1_node_deletion_cost,
    G2_node_insertion_cost,
    G1_edge_deletion_cost,
    G2_edge_insertion_cost,
    algorithm="astar",
)

print(distance)  # 0.5


# Approximate edit distance with beam search, it is more than or equal to the exact edit distance
distance, node_mapping, edge_mapping = graph_edit_distance(
    G1, G2, algorithm="beam", max_beam_size=2
)
print(distance)  # 3.0

# Approximate edit distance with bipartite heuristic, it is more than or equal to the exact edit distance
distance, node_mapping, edge_mapping = graph_edit_distance(
    G1, G2, algorithm="bipartite"
)
print(
    distance
)  # 9.0, can be different as multiple solutions possible for the intermediate LAP used in this approximation


# Approximate edit distance with hausdorff heuristic, it is less than or equal to the exact edit distance
distance, node_mapping, edge_mapping = graph_edit_distance(
    G1, G2, algorithm="hausdorff"
)
print(distance)  # 0.0
