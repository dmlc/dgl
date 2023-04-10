from copy import deepcopy
from heapq import heapify, heappop, heappush, nsmallest

import dgl
import numpy as np

# We use lapjv implementation (https://github.com/src-d/lapjv) to solve assignment problem, because of its scalability
# Also see https://github.com/berhane/LAP-solvers for benchmarking of LAP solvers
from lapjv import lapjv

EPSILON = 0.0000001


def validate_cost_functions(
    G1,
    G2,
    node_substitution_cost=None,
    edge_substitution_cost=None,
    G1_node_deletion_cost=None,
    G1_edge_deletion_cost=None,
    G2_node_insertion_cost=None,
    G2_edge_insertion_cost=None,
):
    """Validates cost functions (substitution, insertion, deletion) and initializes them with default=0 for substitution
    and default=1 for insertion/deletion
    if the provided ones are None.


    Parameters : see graph_edit_distance

    """
    num_G1_nodes = G1.num_nodes()
    num_G2_nodes = G2.num_nodes()

    num_G1_edges = G1.num_edges()
    num_G2_edges = G2.num_edges()

    # if any cost matrix is None, initialize it with default costs
    if node_substitution_cost is None:
        node_substitution_cost = np.zeros(
            (num_G1_nodes, num_G2_nodes), dtype=float
        )
    else:
        assert node_substitution_cost.shape == (num_G1_nodes, num_G2_nodes)

    if edge_substitution_cost is None:
        edge_substitution_cost = np.zeros(
            (num_G1_edges, num_G2_edges), dtype=float
        )
    else:
        assert edge_substitution_cost.shape == (num_G1_edges, num_G2_edges)

    if G1_node_deletion_cost is None:
        G1_node_deletion_cost = np.ones(num_G1_nodes, dtype=float)
    else:
        assert G1_node_deletion_cost.shape[0] == num_G1_nodes

    if G1_edge_deletion_cost is None:
        G1_edge_deletion_cost = np.ones(num_G1_edges, dtype=float)
    else:
        assert G1_edge_deletion_cost.shape[0] == num_G1_edges

    if G2_node_insertion_cost is None:
        G2_node_insertion_cost = np.ones(num_G2_nodes, dtype=float)
    else:
        assert G2_node_insertion_cost.shape[0] == num_G2_nodes

    if G2_edge_insertion_cost is None:
        G2_edge_insertion_cost = np.ones(num_G2_edges, dtype=float)
    else:
        assert G2_edge_insertion_cost.shape[0] == num_G2_edges

    return (
        node_substitution_cost,
        edge_substitution_cost,
        G1_node_deletion_cost,
        G1_edge_deletion_cost,
        G2_node_insertion_cost,
        G2_edge_insertion_cost,
    )


def construct_cost_functions(
    G1,
    G2,
    node_substitution_cost,
    edge_substitution_cost,
    G1_node_deletion_cost,
    G1_edge_deletion_cost,
    G2_node_insertion_cost,
    G2_edge_insertion_cost,
):
    """Constructs cost matrices for LAP solution


    Parameters : see graph_edit_distance

    """
    num_G1_nodes = G1.num_nodes()
    num_G2_nodes = G2.num_nodes()

    num_G1_edges = G1.num_edges()
    num_G2_edges = G2.num_edges()

    # cost matrix of node mappings
    cost_upper_bound = (
        node_substitution_cost.sum()
        + G1_node_deletion_cost.sum()
        + G2_node_insertion_cost.sum()
        + 1
    )
    C_node = np.zeros(
        (num_G1_nodes + num_G2_nodes, num_G1_nodes + num_G2_nodes), dtype=float
    )

    C_node[0:num_G1_nodes, 0:num_G2_nodes] = node_substitution_cost
    C_node[
        0:num_G1_nodes, num_G2_nodes : num_G2_nodes + num_G1_nodes
    ] = np.array(
        [
            G1_node_deletion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G1_nodes)
            for j in range(num_G1_nodes)
        ]
    ).reshape(
        num_G1_nodes, num_G1_nodes
    )
    C_node[
        num_G1_nodes : num_G1_nodes + num_G2_nodes, 0:num_G2_nodes
    ] = np.array(
        [
            G2_node_insertion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G2_nodes)
            for j in range(num_G2_nodes)
        ]
    ).reshape(
        num_G2_nodes, num_G2_nodes
    )

    # cost matrix of edge mappings
    cost_upper_bound = (
        edge_substitution_cost.sum()
        + G1_edge_deletion_cost.sum()
        + G2_edge_insertion_cost.sum()
        + 1
    )
    C_edge = np.zeros(
        (num_G1_edges + num_G2_edges, num_G1_edges + num_G2_edges), dtype=float
    )

    C_edge[0:num_G1_edges, 0:num_G2_edges] = edge_substitution_cost
    C_edge[
        0:num_G1_edges, num_G2_edges : num_G2_edges + num_G1_edges
    ] = np.array(
        [
            G1_edge_deletion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G1_edges)
            for j in range(num_G1_edges)
        ]
    ).reshape(
        num_G1_edges, num_G1_edges
    )
    C_edge[
        num_G1_edges : num_G1_edges + num_G2_edges, 0:num_G2_edges
    ] = np.array(
        [
            G2_edge_insertion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G2_edges)
            for j in range(num_G2_edges)
        ]
    ).reshape(
        num_G2_edges, num_G2_edges
    )
    return C_node, C_edge


def get_edges_to_match(G, node_id, matched_nodes):
    # Find the edges in G with one end-point as node_id and other in matched_nodes or node_id
    incident_edges = np.array([], dtype=int)
    index = np.array([], dtype=int)
    direction = np.array([], dtype=int)
    if G.has_edge_between(node_id, node_id):
        self_edge_ids = G.edge_ids(node_id, node_id, return_array=True).numpy()
        incident_edges = np.concatenate((incident_edges, self_edge_ids))
        index = np.concatenate((index, [-1] * len(self_edge_ids)))
        direction = np.concatenate((direction, [0] * len(self_edge_ids)))
    # Find predecessors
    src, _, eid = G.in_edges([node_id], "all")
    eid = eid.numpy()
    src = src.numpy()
    filtered_indices = [
        (i, matched_nodes.index(src[i]))
        for i in range(len(src))
        if src[i] in matched_nodes
    ]
    matched_index = np.array([_[1] for _ in filtered_indices], dtype=int)
    eid_index = np.array([_[0] for _ in filtered_indices], dtype=int)
    index = np.concatenate((index, matched_index))
    incident_edges = np.concatenate((incident_edges, eid[eid_index]))
    direction = np.concatenate(
        (direction, np.array([-1] * len(filtered_indices), dtype=int))
    )
    # Find successors
    _, dst, eid = G.out_edges([node_id], "all")
    eid = eid.numpy()
    dst = dst.numpy()
    filtered_indices = [
        (i, matched_nodes.index(dst[i]))
        for i in range(len(dst))
        if dst[i] in matched_nodes
    ]
    matched_index = np.array([_[1] for _ in filtered_indices], dtype=int)
    eid_index = np.array([_[0] for _ in filtered_indices], dtype=int)
    index = np.concatenate((index, matched_index))
    incident_edges = np.concatenate((incident_edges, eid[eid_index]))
    direction = np.concatenate(
        (direction, np.array([1] * len(filtered_indices), dtype=int))
    )
    return incident_edges, index, direction


def subset_cost_matrix(cost_matrix, row_ids, col_ids, num_rows, num_cols):
    # Extract thr subset of cost matrix corresponding to rows/cols in arrays row_ids/col_ids
    # Note that the shape of cost_matrix is (num_rows+num_cols) * (num_rows+num_cols)
    extended_row_ids = np.concatenate(
        (row_ids, np.array([k + num_rows for k in col_ids]))
    )
    extended_col_ids = np.concatenate(
        (col_ids, np.array([k + num_cols for k in row_ids]))
    )
    return cost_matrix[extended_row_ids, :][:, extended_col_ids]


class search_tree_node:
    def __init__(
        self,
        G1,
        G2,
        parent_matched_cost,
        parent_matched_nodes,
        parent_matched_edges,
        node_G1,
        node_G2,
        parent_unprocessed_nodes_G1,
        parent_unprocessed_nodes_G2,
        parent_unprocessed_edges_G1,
        parent_unprocessed_edges_G2,
        cost_matrix_nodes,
        cost_matrix_edges,
    ):
        self.matched_cost = parent_matched_cost
        self.future_approximate_cost = 0.0
        self.matched_nodes = deepcopy(parent_matched_nodes)
        self.matched_nodes[0].append(node_G1)
        self.matched_nodes[1].append(node_G2)
        self.matched_edges = deepcopy(parent_matched_edges)
        self.unprocessed_nodes_G1 = [
            _ for _ in parent_unprocessed_nodes_G1 if _ != node_G1
        ]
        self.unprocessed_nodes_G2 = [
            _ for _ in parent_unprocessed_nodes_G2 if _ != node_G2
        ]

        # Add the cost of matching nodes at this tree-node to the matched cost
        if (
            node_G1 is not None and node_G2 is not None
        ):  # Substitute node_G1 with node_G2
            self.matched_cost += cost_matrix_nodes[node_G1, node_G2]
        elif node_G1 is not None:  # Delete node_G1
            self.matched_cost += cost_matrix_nodes[
                node_G1, node_G1 + G2.num_nodes()
            ]
        elif node_G2 is not None:  # Insert node_G2
            self.matched_cost += cost_matrix_nodes[
                node_G2 + G1.num_nodes(), node_G2
            ]

        # Add the cost of matching edges at this tree-node to the matched cost
        incident_edges_G1 = []
        if (
            node_G1 is not None
        ):  # Find the edges with one end-point as node_G1 and other in matched nodes or node_G1
            incident_edges_G1, index_G1, direction_G1 = get_edges_to_match(
                G1, node_G1, parent_matched_nodes[0]
            )

        incident_edges_G2 = np.array([])
        if (
            node_G2 is not None
        ):  # Find the edges with one end-point as node_G2 and other in matched nodes or node_G2
            incident_edges_G2, index_G2, direction_G2 = get_edges_to_match(
                G2, node_G2, parent_matched_nodes[1]
            )

        if (
            len(incident_edges_G1) > 0 and len(incident_edges_G2) > 0
        ):  # Consider substituting
            matched_edges_cost_matrix = subset_cost_matrix(
                cost_matrix_edges,
                incident_edges_G1,
                incident_edges_G2,
                G1.num_edges(),
                G2.num_edges(),
            )
            max_sum = matched_edges_cost_matrix.sum()
            # take care of impossible assignments by assigning maximum cost
            for i in range(len(incident_edges_G1)):
                for j in range(len(incident_edges_G2)):
                    # both edges need to have same direction and the other end nodes are matched
                    if (
                        direction_G1[i] == direction_G2[j]
                        and index_G1[i] == index_G2[j]
                    ):
                        continue
                    else:
                        matched_edges_cost_matrix[i, j] = max_sum
            # Match the edges as per the LAP solution
            row_ind, col_ind, _ = lapjv(matched_edges_cost_matrix)
            lap_cost = 0.00
            for i in range(len(row_ind)):
                lap_cost += matched_edges_cost_matrix[i, row_ind[i]]

            # Update matched edges
            for i in range(len(row_ind)):
                if i < len(incident_edges_G1):
                    self.matched_edges[0].append(incident_edges_G1[i])
                    if row_ind[i] < len(incident_edges_G2):
                        self.matched_edges[1].append(
                            incident_edges_G2[row_ind[i]]
                        )
                    else:
                        self.matched_edges[1].append(None)
                elif row_ind[i] < len(incident_edges_G2):
                    self.matched_edges[0].append(None)
                    self.matched_edges[1].append(incident_edges_G2[row_ind[i]])
            self.matched_cost += lap_cost

        elif len(incident_edges_G1) > 0:  # only deletion possible
            edge_deletion_cost = 0.0
            for edge in incident_edges_G1:
                edge_deletion_cost += cost_matrix_edges[
                    edge, G2.num_edges() + edge
                ]
            # Update matched edges
            for edge in incident_edges_G1:
                self.matched_edges[0].append(edge)
                self.matched_edges[1].append(None)

                # Update matched edges

            self.matched_cost += edge_deletion_cost

        elif len(incident_edges_G2) > 0:  # only insertion possible
            edge_insertion_cost = 0.0
            for edge in incident_edges_G2:
                edge_insertion_cost += cost_matrix_edges[
                    G1.num_edges() + edge, edge
                ]
            # Update matched edges
            for edge in incident_edges_G2:
                self.matched_edges[0].append(None)
                self.matched_edges[1].append(edge)

            self.matched_cost += edge_insertion_cost

        # Add the cost of matching of unprocessed nodes to the future approximate cost
        if (
            len(self.unprocessed_nodes_G1) > 0
            and len(self.unprocessed_nodes_G2) > 0
        ):  # Consider substituting
            unmatched_nodes_cost_matrix = subset_cost_matrix(
                cost_matrix_nodes,
                self.unprocessed_nodes_G1,
                self.unprocessed_nodes_G2,
                G1.num_nodes(),
                G2.num_nodes(),
            )
            # Match the edges as per the LAP solution
            row_ind, col_ind, _ = lapjv(unmatched_nodes_cost_matrix)
            lap_cost = 0.00
            for i in range(len(row_ind)):
                lap_cost += unmatched_nodes_cost_matrix[i, row_ind[i]]

            self.future_approximate_cost += lap_cost

        elif len(self.unprocessed_nodes_G1) > 0:  # only deletion possible
            node_deletion_cost = 0.0
            for node in self.unprocessed_nodes_G1:
                node_deletion_cost += cost_matrix_nodes[
                    node, G2.num_nodes() + node
                ]

            self.future_approximate_cost += node_deletion_cost

        elif len(self.unprocessed_nodes_G2) > 0:  # only insertion possible
            node_insertion_cost = 0.0
            for node in self.unprocessed_nodes_G2:
                node_insertion_cost += cost_matrix_nodes[
                    G1.num_nodes() + node, node
                ]

            self.future_approximate_cost += node_insertion_cost

        # Add the cost of LAP matching of unprocessed edges to the future approximate cost
        self.unprocessed_edges_G1 = [
            _ for _ in parent_unprocessed_edges_G1 if _ not in incident_edges_G1
        ]
        self.unprocessed_edges_G2 = [
            _ for _ in parent_unprocessed_edges_G2 if _ not in incident_edges_G2
        ]
        if (
            len(self.unprocessed_edges_G1) > 0
            and len(self.unprocessed_edges_G2) > 0
        ):  # Consider substituting
            unmatched_edges_cost_matrix = subset_cost_matrix(
                cost_matrix_edges,
                self.unprocessed_edges_G1,
                self.unprocessed_edges_G2,
                G1.num_edges(),
                G2.num_edges(),
            )
            # Match the edges as per the LAP solution
            row_ind, col_ind, _ = lapjv(unmatched_edges_cost_matrix)
            lap_cost = 0.00
            for i in range(len(row_ind)):
                lap_cost += unmatched_edges_cost_matrix[i, row_ind[i]]

            self.future_approximate_cost += lap_cost

        elif len(self.unprocessed_edges_G1) > 0:  # only deletion possible
            edge_deletion_cost = 0.0
            for edge in self.unprocessed_edges_G1:
                edge_deletion_cost += cost_matrix_edges[
                    edge, G2.num_edges() + edge
                ]

            self.future_approximate_cost += edge_deletion_cost

        elif len(self.unprocessed_edges_G2) > 0:  # only insertion possible
            edge_insertion_cost = 0.0
            for edge in self.unprocessed_edges_G2:
                edge_insertion_cost += cost_matrix_edges[
                    G1.num_edges() + edge, edge
                ]

            self.future_approximate_cost += edge_insertion_cost

    # For heap insertion order
    def __lt__(self, other):
        if (
            abs(
                (self.matched_cost + self.future_approximate_cost)
                - (other.matched_cost + other.future_approximate_cost)
            )
            > EPSILON
        ):
            return (self.matched_cost + self.future_approximate_cost) < (
                other.matched_cost + other.future_approximate_cost
            )
        elif abs(self.matched_cost - other.matched_cost) > EPSILON:
            return other.matched_cost < self.matched_cost
            # matched cost is closer to reality
        else:
            return (
                len(self.unprocessed_nodes_G1)
                + len(self.unprocessed_nodes_G2)
                + len(self.unprocessed_edges_G1)
                + len(self.unprocessed_edges_G2)
            ) < (
                len(other.unprocessed_nodes_G1)
                + len(other.unprocessed_nodes_G2)
                + len(other.unprocessed_edges_G1)
                + len(other.unprocessed_edges_G2)
            )


def edit_cost_from_node_matching(
    G1, G2, cost_matrix_nodes, cost_matrix_edges, node_matching
):
    matched_cost = 0.0
    matched_nodes = ([], [])
    matched_edges = ([], [])
    # Add the cost of matching nodes
    for i in range(G1.num_nodes()):
        matched_cost += cost_matrix_nodes[i, node_matching[i]]
        matched_nodes[0].append(i)
        if node_matching[i] < G2.num_nodes():
            matched_nodes[1].append(node_matching[i])
        else:
            matched_nodes[1].append(None)
    for i in range(G1.num_nodes(), len(node_matching)):
        matched_cost += cost_matrix_nodes[i, node_matching[i]]
        if node_matching[i] < G2.num_nodes():
            matched_nodes[0].append(None)
            matched_nodes[1].append(node_matching[i])

    for i in range(len(matched_nodes[0])):
        # Add the cost of matching edges
        incident_edges_G1 = []
        if (
            matched_nodes[0][i] is not None
        ):  # Find the edges with one end-point as node_G1 and other in matched nodes or node_G1
            incident_edges_G1, index_G1, direction_G1 = get_edges_to_match(
                G1, matched_nodes[0][i], matched_nodes[0][:i]
            )

        incident_edges_G2 = np.array([])
        if (
            matched_nodes[1][i] is not None
        ):  # Find the edges with one end-point as node_G2 and other in matched nodes or node_G2
            incident_edges_G2, index_G2, direction_G2 = get_edges_to_match(
                G2, matched_nodes[1][i], matched_nodes[1][:i]
            )

        if (
            len(incident_edges_G1) > 0 and len(incident_edges_G2) > 0
        ):  # Consider substituting
            matched_edges_cost_matrix = subset_cost_matrix(
                cost_matrix_edges,
                incident_edges_G1,
                incident_edges_G2,
                G1.num_edges(),
                G2.num_edges(),
            )
            max_sum = matched_edges_cost_matrix.sum()
            # take care of impossible assignments by assigning maximum cost
            for i in range(len(incident_edges_G1)):
                for j in range(len(incident_edges_G2)):
                    # both edges need to have same direction and the other end nodes are matched
                    if (
                        direction_G1[i] == direction_G2[j]
                        and index_G1[i] == index_G2[j]
                    ):
                        continue
                    else:
                        matched_edges_cost_matrix[i, j] = max_sum
            # Match the edges as per the LAP solution
            row_ind, col_ind, _ = lapjv(matched_edges_cost_matrix)
            lap_cost = 0.00
            for i in range(len(row_ind)):
                lap_cost += matched_edges_cost_matrix[i, row_ind[i]]

            # Update matched edges
            for i in range(len(row_ind)):
                if i < len(incident_edges_G1):
                    matched_edges[0].append(incident_edges_G1[i])
                    if row_ind[i] < len(incident_edges_G2):
                        matched_edges[1].append(incident_edges_G2[row_ind[i]])
                    else:
                        matched_edges[1].append(None)
                elif row_ind[i] < len(incident_edges_G2):
                    matched_edges[0].append(None)
                    matched_edges[1].append(incident_edges_G2[row_ind[i]])
            matched_cost += lap_cost

        elif len(incident_edges_G1) > 0:  # only deletion possible
            edge_deletion_cost = 0.0
            for edge in incident_edges_G1:
                edge_deletion_cost += cost_matrix_edges[
                    edge, G2.num_edges() + edge
                ]
            # Update matched edges
            for edge in incident_edges_G1:
                matched_edges[0].append(edge)
                matched_edges[1].append(None)

                # Update matched edges

            matched_cost += edge_deletion_cost

        elif len(incident_edges_G2) > 0:  # only insertion possible
            edge_insertion_cost = 0.0
            for edge in incident_edges_G2:
                edge_insertion_cost += cost_matrix_edges[
                    G1.num_edges() + edge, edge
                ]
            # Update matched edges
            for edge in incident_edges_G2:
                matched_edges[0].append(None)
                matched_edges[1].append(edge)

            matched_cost += edge_insertion_cost

    return (matched_cost, matched_nodes, matched_edges)


def contextual_cost_matrix_construction(
    G1,
    G2,
    node_substitution_cost,
    edge_substitution_cost,
    G1_node_deletion_cost,
    G1_edge_deletion_cost,
    G2_node_insertion_cost,
    G2_edge_insertion_cost,
):
    # Calculates approximate GED using linear assignment on the nodes with bipartite algorithm
    # cost matrix of node mappings

    num_G1_nodes = G1.num_nodes()
    num_G2_nodes = G2.num_nodes()

    num_G1_edges = G1.num_edges()
    num_G2_edges = G2.num_edges()

    cost_upper_bound = 2 * (
        node_substitution_cost.sum()
        + G1_node_deletion_cost.sum()
        + G2_node_insertion_cost.sum()
        + 1
    )
    cost_matrix = np.zeros(
        (num_G1_nodes + num_G2_nodes, num_G1_nodes + num_G2_nodes), dtype=float
    )

    cost_matrix[0:num_G1_nodes, 0:num_G2_nodes] = node_substitution_cost
    cost_matrix[
        0:num_G1_nodes, num_G2_nodes : num_G2_nodes + num_G1_nodes
    ] = np.array(
        [
            G1_node_deletion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G1_nodes)
            for j in range(num_G1_nodes)
        ]
    ).reshape(
        num_G1_nodes, num_G1_nodes
    )
    cost_matrix[
        num_G1_nodes : num_G1_nodes + num_G2_nodes, 0:num_G2_nodes
    ] = np.array(
        [
            G2_node_insertion_cost[i] if i == j else cost_upper_bound
            for i in range(num_G2_nodes)
            for j in range(num_G2_nodes)
        ]
    ).reshape(
        num_G2_nodes, num_G2_nodes
    )

    self_edge_list_G1 = [np.array([], dtype=int)] * num_G1_nodes
    self_edge_list_G2 = [np.array([], dtype=int)] * num_G2_nodes
    incoming_edges_G1 = [np.array([], dtype=int)] * num_G1_nodes
    incoming_edges_G2 = [np.array([], dtype=int)] * num_G2_nodes
    outgoing_edges_G1 = [np.array([], dtype=int)] * num_G1_nodes
    outgoing_edges_G2 = [np.array([], dtype=int)] * num_G2_nodes

    for i in range(num_G1_nodes):
        if G1.has_edge_between(i, i):
            self_edge_list_G1[i] = sorted(
                G1.edge_ids(i, i, return_array=True).numpy()
            )
        incoming_edges_G1[i] = G1.in_edges([i], "eid").numpy()
        incoming_edges_G1[i] = np.setdiff1d(
            incoming_edges_G1[i], self_edge_list_G1[i]
        )
        outgoing_edges_G1[i] = G1.out_edges([i], "eid").numpy()
        outgoing_edges_G1[i] = np.setdiff1d(
            outgoing_edges_G1[i], self_edge_list_G1[i]
        )
    for i in range(num_G2_nodes):
        if G2.has_edge_between(i, i):
            self_edge_list_G2[i] = sorted(
                G2.edge_ids(i, i, return_array=True).numpy()
            )
        incoming_edges_G2[i] = G2.in_edges([i], "eid").numpy()
        incoming_edges_G2[i] = np.setdiff1d(
            incoming_edges_G2[i], self_edge_list_G2[i]
        )
        outgoing_edges_G2[i] = G2.out_edges([i], "eid").numpy()
        outgoing_edges_G2[i] = np.setdiff1d(
            outgoing_edges_G2[i], self_edge_list_G2[i]
        )

    selected_deletion_G1 = [
        G1_edge_deletion_cost[
            np.concatenate(
                (
                    self_edge_list_G1[i],
                    incoming_edges_G1[i],
                    outgoing_edges_G1[i],
                )
            )
        ]
        for i in range(G1.num_nodes())
    ]
    selected_insertion_G2 = [
        G2_edge_insertion_cost[
            np.concatenate(
                (
                    self_edge_list_G2[i],
                    incoming_edges_G2[i],
                    outgoing_edges_G2[i],
                )
            )
        ]
        for i in range(G2.num_nodes())
    ]

    # Add the cost of edge edition which are dependent of a node (see this as the cost associated with a substructure)
    for i in range(num_G1_nodes):
        for j in range(num_G2_nodes):
            m = (
                len(self_edge_list_G1[i])
                + len(incoming_edges_G1[i])
                + len(outgoing_edges_G1[i])
            )
            n = (
                len(self_edge_list_G2[j])
                + len(incoming_edges_G2[j])
                + len(outgoing_edges_G2[j])
            )

            matrix_dim = m + n

            if matrix_dim == 0:
                continue
            temp_edge_cost_matrix = np.empty((matrix_dim, matrix_dim))
            temp_edge_cost_matrix.fill(cost_upper_bound)

            temp_edge_cost_matrix[
                : len(self_edge_list_G1[i]), : len(self_edge_list_G2[j])
            ] = edge_substitution_cost[self_edge_list_G1[i], :][
                :, self_edge_list_G2[j]
            ]
            temp_edge_cost_matrix[
                len(self_edge_list_G1[i]) : len(self_edge_list_G1[i])
                + len(incoming_edges_G1[i]),
                len(self_edge_list_G2[j]) : len(self_edge_list_G2[j])
                + len(incoming_edges_G2[j]),
            ] = edge_substitution_cost[incoming_edges_G1[i], :][
                :, incoming_edges_G2[j]
            ]
            temp_edge_cost_matrix[
                len(self_edge_list_G1[i]) + len(incoming_edges_G1[i]) : m,
                len(self_edge_list_G2[j]) + len(incoming_edges_G2[j]) : n,
            ] = edge_substitution_cost[outgoing_edges_G1[i], :][
                :, outgoing_edges_G2[j]
            ]

            np.fill_diagonal(
                temp_edge_cost_matrix[:m, n:], selected_deletion_G1[i]
            )
            np.fill_diagonal(
                temp_edge_cost_matrix[m:, :n], selected_insertion_G2[j]
            )

            temp_edge_cost_matrix[m:, n:].fill(0)
            row_ind, col_ind, _ = lapjv(temp_edge_cost_matrix)
            lap_cost = 0.00
            for k in range(len(row_ind)):
                lap_cost += temp_edge_cost_matrix[k, row_ind[k]]

            cost_matrix[i, j] += lap_cost

    for i in range(num_G1_nodes):
        cost_matrix[i, num_G2_nodes + i] += selected_deletion_G1[i].sum()

    for i in range(num_G2_nodes):
        cost_matrix[num_G1_nodes + i, i] += selected_insertion_G2[i].sum()

    return cost_matrix


def hausdorff_matching(
    G1,
    G2,
    node_substitution_cost,
    edge_substitution_cost,
    G1_node_deletion_cost,
    G1_edge_deletion_cost,
    G2_node_insertion_cost,
    G2_edge_insertion_cost,
):
    # Calculates approximate GED using hausdorff_matching
    # cost matrix of node mappings

    num_G1_nodes = G1.num_nodes()
    num_G2_nodes = G2.num_nodes()

    num_G1_edges = G1.num_edges()
    num_G2_edges = G2.num_edges()

    self_edge_list_G1 = [np.array([], dtype=int)] * num_G1_nodes
    self_edge_list_G2 = [np.array([], dtype=int)] * num_G2_nodes
    incoming_edges_G1 = [np.array([], dtype=int)] * num_G1_nodes
    incoming_edges_G2 = [np.array([], dtype=int)] * num_G2_nodes
    outgoing_edges_G1 = [np.array([], dtype=int)] * num_G1_nodes
    outgoing_edges_G2 = [np.array([], dtype=int)] * num_G2_nodes

    for i in range(num_G1_nodes):
        if G1.has_edge_between(i, i):
            self_edge_list_G1[i] = sorted(
                G1.edge_ids(i, i, return_array=True).numpy()
            )
        incoming_edges_G1[i] = G1.in_edges([i], "eid").numpy()
        incoming_edges_G1[i] = np.setdiff1d(
            incoming_edges_G1[i], self_edge_list_G1[i]
        )
        outgoing_edges_G1[i] = G1.out_edges([i], "eid").numpy()
        outgoing_edges_G1[i] = np.setdiff1d(
            outgoing_edges_G1[i], self_edge_list_G1[i]
        )
    for i in range(num_G2_nodes):
        if G2.has_edge_between(i, i):
            self_edge_list_G2[i] = sorted(
                G2.edge_ids(i, i, return_array=True).numpy()
            )
        incoming_edges_G2[i] = G2.in_edges([i], "eid").numpy()
        incoming_edges_G2[i] = np.setdiff1d(
            incoming_edges_G2[i], self_edge_list_G2[i]
        )
        outgoing_edges_G2[i] = G2.out_edges([i], "eid").numpy()
        outgoing_edges_G2[i] = np.setdiff1d(
            outgoing_edges_G2[i], self_edge_list_G2[i]
        )

    selected_deletion_self_G1 = [
        G1_edge_deletion_cost[self_edge_list_G1[i]]
        for i in range(G1.num_nodes())
    ]
    selected_insertion_self_G2 = [
        G2_edge_insertion_cost[self_edge_list_G2[i]]
        for i in range(G2.num_nodes())
    ]

    selected_deletion_incoming_G1 = [
        G1_edge_deletion_cost[incoming_edges_G1[i]]
        for i in range(G1.num_nodes())
    ]
    selected_insertion_incoming_G2 = [
        G2_edge_insertion_cost[incoming_edges_G2[i]]
        for i in range(G2.num_nodes())
    ]

    selected_deletion_outgoing_G1 = [
        G1_edge_deletion_cost[outgoing_edges_G1[i]]
        for i in range(G1.num_nodes())
    ]
    selected_insertion_outgoing_G2 = [
        G2_edge_insertion_cost[outgoing_edges_G2[i]]
        for i in range(G2.num_nodes())
    ]

    selected_deletion_G1 = [
        G1_edge_deletion_cost[
            np.concatenate(
                (
                    self_edge_list_G1[i],
                    incoming_edges_G1[i],
                    outgoing_edges_G1[i],
                )
            )
        ]
        for i in range(G1.num_nodes())
    ]
    selected_insertion_G2 = [
        G2_edge_insertion_cost[
            np.concatenate(
                (
                    self_edge_list_G2[i],
                    incoming_edges_G2[i],
                    outgoing_edges_G2[i],
                )
            )
        ]
        for i in range(G2.num_nodes())
    ]

    cost_G1 = np.array(
        [
            (G1_node_deletion_cost[i] + selected_deletion_G1[i].sum() / 2)
            for i in range(num_G1_nodes)
        ]
    )
    cost_G2 = np.array(
        [
            (G2_node_insertion_cost[i] + selected_insertion_G2[i].sum() / 2)
            for i in range(num_G2_nodes)
        ]
    )

    for i in range(num_G1_nodes):
        for j in range(num_G2_nodes):
            c1_self = deepcopy(selected_deletion_self_G1[i])
            c2_self = deepcopy(selected_insertion_self_G2[j])
            c1_incoming = deepcopy(selected_deletion_incoming_G1[i])
            c2_incoming = deepcopy(selected_insertion_incoming_G2[j])
            c1_outgoing = deepcopy(selected_deletion_outgoing_G1[i])
            c2_outgoing = deepcopy(selected_insertion_outgoing_G2[j])

            for k, a in enumerate(self_edge_list_G1[i]):
                for l, b in enumerate(self_edge_list_G2[j]):
                    c1_self[k] = min(
                        c1_self[k], edge_substitution_cost[a, b] / 2
                    )
                    c2_self[l] = min(
                        c2_self[l], edge_substitution_cost[a, b] / 2
                    )

            for k, a in enumerate(incoming_edges_G1[i]):
                for l, b in enumerate(incoming_edges_G2[j]):
                    c1_incoming[k] = min(
                        c1_incoming[k], edge_substitution_cost[a, b] / 2
                    )
                    c2_incoming[l] = min(
                        c2_incoming[l], edge_substitution_cost[a, b] / 2
                    )

            for k, a in enumerate(outgoing_edges_G1[i]):
                for l, b in enumerate(outgoing_edges_G2[j]):
                    c1_outgoing[k] = min(
                        c1_outgoing[k], edge_substitution_cost[a, b] / 2
                    )
                    c2_outgoing[l] = min(
                        c2_outgoing[l], edge_substitution_cost[a, b] / 2
                    )

            edge_hausdorff_lower_bound = 0.0

            if len(selected_deletion_G1[i]) > len(selected_insertion_G2[j]):
                idx = np.argpartition(
                    selected_deletion_G1[i],
                    (
                        len(selected_deletion_G1[i])
                        - len(selected_insertion_G2[j])
                    ),
                )
                edge_hausdorff_lower_bound = selected_deletion_G1[i][
                    idx[
                        : (
                            len(selected_deletion_G1[i])
                            - len(selected_insertion_G2[j])
                        )
                    ]
                ].sum()
            elif len(selected_deletion_G1[i]) < len(selected_insertion_G2[j]):
                idx = np.argpartition(
                    selected_insertion_G2[j],
                    (
                        len(selected_insertion_G2[j])
                        - len(selected_deletion_G1[i])
                    ),
                )
                edge_hausdorff_lower_bound = selected_insertion_G2[j][
                    idx[
                        : (
                            len(selected_insertion_G2[j])
                            - len(selected_deletion_G1[i])
                        )
                    ]
                ].sum()

            sc_cost = 0.5 * (
                node_substitution_cost[i, j]
                + 0.5
                * max(
                    c1_self.sum()
                    + c2_self.sum()
                    + c1_incoming.sum()
                    + c2_incoming.sum()
                    + c1_outgoing.sum()
                    + c2_outgoing.sum(),
                    edge_hausdorff_lower_bound,
                )
            )

            if cost_G1[i] > sc_cost:
                cost_G1[i] = sc_cost
            if cost_G2[j] > sc_cost:
                cost_G2[j] = sc_cost

    graph_hausdorff_lower_bound = 0.0
    if num_G1_nodes > num_G2_nodes:
        idx = np.argpartition(
            G1_node_deletion_cost, (num_G1_nodes - num_G2_nodes)
        )
        graph_hausdorff_lower_bound = G1_node_deletion_cost[
            idx[: (num_G1_nodes - num_G2_nodes)]
        ].sum()
    elif num_G1_nodes < num_G2_nodes:
        idx = np.argpartition(
            G2_node_insertion_cost, (num_G2_nodes - num_G1_nodes)
        )
        graph_hausdorff_lower_bound = G2_node_insertion_cost[
            idx[: (num_G2_nodes - num_G1_nodes)]
        ].sum()

    graph_hausdorff_cost = max(
        graph_hausdorff_lower_bound, cost_G1.sum() + cost_G2.sum()
    )
    return graph_hausdorff_cost


def a_star_search(G1, G2, cost_matrix_nodes, cost_matrix_edges, max_beam_size):
    # A-star traversal
    open_list = []
    # Create first nodes in the A-star search tree, matching node 0 of G1 with all possibilities (each node of G2, and deletion)
    matched_cost = 0.0
    matched_nodes = ([], [])
    # No nodes matched in the beginning
    matched_edges = ([], [])
    # No edges matched in the beginning
    unprocessed_nodes_G1 = [
        i for i in range(G1.num_nodes())
    ]  # No nodes matched in the beginning
    unprocessed_nodes_G2 = [
        i for i in range(G2.num_nodes())
    ]  # No nodes matched in the beginning
    unprocessed_edges_G1 = [
        i for i in range(G1.num_edges())
    ]  # No edges matched in the beginning
    unprocessed_edges_G2 = [
        i for i in range(G2.num_edges())
    ]  # No edges matched in the beginning

    for i in range(len(unprocessed_nodes_G2)):
        tree_node = search_tree_node(
            G1,
            G2,
            matched_cost,
            matched_nodes,
            matched_edges,
            unprocessed_nodes_G1[0],
            unprocessed_nodes_G2[i],
            unprocessed_nodes_G1,
            unprocessed_nodes_G2,
            unprocessed_edges_G1,
            unprocessed_edges_G2,
            cost_matrix_nodes,
            cost_matrix_edges,
        )
        # Insert into open-list, implemented as a heap

        heappush(open_list, tree_node)

    # Consider node deletion
    tree_node = search_tree_node(
        G1,
        G2,
        matched_cost,
        matched_nodes,
        matched_edges,
        unprocessed_nodes_G1[0],
        None,
        unprocessed_nodes_G1,
        unprocessed_nodes_G2,
        unprocessed_edges_G1,
        unprocessed_edges_G2,
        cost_matrix_nodes,
        cost_matrix_edges,
    )
    # Insert into open-list, implemented as a heap
    heappush(open_list, tree_node)

    while len(open_list) > 0:
        # TODO: Create a node that processes multi node insertion deletion in one search node,
        # as opposed in multiple search nodes here
        parent_tree_node = heappop(open_list)
        matched_cost = parent_tree_node.matched_cost
        matched_nodes = parent_tree_node.matched_nodes
        matched_edges = parent_tree_node.matched_edges
        unprocessed_nodes_G1 = parent_tree_node.unprocessed_nodes_G1
        unprocessed_nodes_G2 = parent_tree_node.unprocessed_nodes_G2
        unprocessed_edges_G1 = parent_tree_node.unprocessed_edges_G1
        unprocessed_edges_G2 = parent_tree_node.unprocessed_edges_G2

        if len(unprocessed_nodes_G1) == 0 and len(unprocessed_nodes_G2) == 0:
            return (matched_cost, matched_nodes, matched_edges)
        elif len(unprocessed_nodes_G1) > 0:
            for i in range(len(unprocessed_nodes_G2)):
                tree_node = search_tree_node(
                    G1,
                    G2,
                    matched_cost,
                    matched_nodes,
                    matched_edges,
                    unprocessed_nodes_G1[0],
                    unprocessed_nodes_G2[i],
                    unprocessed_nodes_G1,
                    unprocessed_nodes_G2,
                    unprocessed_edges_G1,
                    unprocessed_edges_G2,
                    cost_matrix_nodes,
                    cost_matrix_edges,
                )
                # Insert into open-list, implemented as a heap
                heappush(open_list, tree_node)

            # Consider node deletion
            tree_node = search_tree_node(
                G1,
                G2,
                matched_cost,
                matched_nodes,
                matched_edges,
                unprocessed_nodes_G1[0],
                None,
                unprocessed_nodes_G1,
                unprocessed_nodes_G2,
                unprocessed_edges_G1,
                unprocessed_edges_G2,
                cost_matrix_nodes,
                cost_matrix_edges,
            )
            # Insert into open-list, implemented as a heap
            heappush(open_list, tree_node)

        elif len(unprocessed_nodes_G2) > 0:
            for i in range(len(unprocessed_nodes_G2)):
                tree_node = search_tree_node(
                    G1,
                    G2,
                    matched_cost,
                    matched_nodes,
                    matched_edges,
                    None,
                    unprocessed_nodes_G2[i],
                    unprocessed_nodes_G1,
                    unprocessed_nodes_G2,
                    unprocessed_edges_G1,
                    unprocessed_edges_G2,
                    cost_matrix_nodes,
                    cost_matrix_edges,
                )
                # Insert into open-list, implemented as a heap
                heappush(open_list, tree_node)

        # Retain the top-k elements in open-list iff algorithm is beam
        if max_beam_size > 0 and len(open_list) > max_beam_size:
            open_list = nsmallest(max_beam_size, open_list)
            heapify(open_list)

    return None


def get_sorted_mapping(mapping_tuple, len1, len2):
    # Get sorted mapping of nodes/edges
    result_0 = [None] * len1
    result_1 = [None] * len2
    for i in range(len(mapping_tuple[0])):
        if mapping_tuple[0][i] is not None and mapping_tuple[1][i] is not None:
            result_0[mapping_tuple[0][i]] = mapping_tuple[1][i]
            result_1[mapping_tuple[1][i]] = mapping_tuple[0][i]
    return (result_0, result_1)


def graph_edit_distance(
    G1,
    G2,
    node_substitution_cost=None,
    edge_substitution_cost=None,
    G1_node_deletion_cost=None,
    G2_node_insertion_cost=None,
    G1_edge_deletion_cost=None,
    G2_edge_insertion_cost=None,
    algorithm="bipartite",
    max_beam_size=100,
):
    """Returns GED (graph edit distance) between DGLGraphs G1 and G2.


    Parameters
    ----------
    G1, G2: DGLGraphs

    node_substitution_cost, edge_substitution_cost : 2D numpy arrays
        node_substitution_cost[i,j] is the cost of substitution node i of G1 with node j of G2,
        similar definition for edge_substitution_cost. If None, default cost of 0 is used.

    G1_node_deletion_cost, G1_edge_deletion_cost : 1D numpy arrays
        G1_node_deletion_cost[i] is the cost of deletion of node i of G1,
        similar definition for G1_edge_deletion_cost. If None, default cost of 1 is used.

    G2_node_insertion_cost, G2_edge_insertion_cost : 1D numpy arrays
        G2_node_insertion_cost[i] is the cost of insertion of node i of G2,
        similar definition for G2_edge_insertion_cost. If None, default cost of 1 is used.

    algorithm : string
        Algorithm to use to calculate the edit distance.
        For now, 4 algorithms are supported
        i) astar: Calculates exact GED using A* graph traversal algorithm,
        the heuristic used is the one proposed in (Riesen and Bunke, 2009) [1].
        ii) beam: Calculates approximate GED using A* graph traversal algorithm,
        with a maximum number of nodes in the open list. [2]
        iii) bipartite (default): Calculates approximate GED using linear assignment on the nodes,
        with jv (Jonker-Volgerand) algorithm. [3]
        iv) hausdorff: Approximation of graph edit distance based on Hausdorff matching [4].

    max_beam_size : int
        Maximum number of nodes in the open list, in case the algorithm is 'beam'.


    Returns
    -------
    A tuple of three objects: (edit_distance, node_mapping, edge_mapping)
    edit distance is the calculated edit distance (float)
    node_mapping is a tuple of size two, containing the node assignments of the two graphs respectively
    eg., node_mapping[0][i] is the node mapping of node i of graph G1 (None means that the node is deleted)
    Similar definition for the edge_mapping

    For 'hausdorff', node_mapping and edge_mapping are returned as None, as this approximation does not return a unique edit path

    Examples
    --------
    >>> src1 = [0, 1, 2, 3, 4, 5];
    >>> dst1 = [1, 2, 3, 4, 5, 6];
    >>> src2 = [0, 1, 3, 4, 5];
    >>> dst2 = [1, 2, 4, 5, 6];

    >>> G1 = dgl.DGLGraph((src1, dst1))
    >>> G2 = dgl.DGLGraph((src2, dst2))
    >>> distance, node_mapping, edge_mapping = graph_edit_distance(G1, G1, algorithm='astar')
    >>> print(distance)
    0.0
    >>> distance, node_mapping, edge_mapping = graph_edit_distance(G1, G2, algorithm='astar')
    >>> print(distance)
    1.0

    References
    ----------
    [1] Riesen, Kaspar, Stefan Fankhauser, and Horst Bunke.
    "Speeding Up Graph Edit Distance Computation with a Bipartite Heuristic."
    MLG. 2007.
    [2] Neuhaus, Michel, Kaspar Riesen, and Horst Bunke.
    "Fast suboptimal algorithms for the computation of graph edit distance."
    Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR)
    and Structural and Syntactic Pattern Recognition (SSPR). 2006.
    [3] Fankhauser, Stefan, Kaspar Riesen, and Horst Bunke.
    "Speeding up graph edit distance computation through fast bipartite matching."
    International Workshop on Graph-Based Representations in Pattern Recognition. 2011.
    [4] Fischer, Andreas, et al. "A hausdorff heuristic for efficient computation of graph edit distance."
    Joint IAPR International Workshops on Statistical Techniques in Pattern Recognition (SPR)
    and Structural and Syntactic Pattern Recognition (SSPR). 2014.

    """
    # Handle corner cases
    if G1 is None and G2 is None:
        return (0.0, ([], []), ([], []))
    elif G1 is None:
        edit_cost = 0.0

    # Validate
    if algorithm != "beam":
        max_beam_size = -1
    (
        node_substitution_cost,
        edge_substitution_cost,
        G1_node_deletion_cost,
        G1_edge_deletion_cost,
        G2_node_insertion_cost,
        G2_edge_insertion_cost,
    ) = validate_cost_functions(
        G1,
        G2,
        node_substitution_cost,
        edge_substitution_cost,
        G1_node_deletion_cost,
        G1_edge_deletion_cost,
        G2_node_insertion_cost,
        G2_edge_insertion_cost,
    )

    # cost matrices for LAP solution
    cost_matrix_nodes, cost_matrix_edges = construct_cost_functions(
        G1,
        G2,
        node_substitution_cost,
        edge_substitution_cost,
        G1_node_deletion_cost,
        G1_edge_deletion_cost,
        G2_node_insertion_cost,
        G2_edge_insertion_cost,
    )

    if algorithm == "astar" or algorithm == "beam":
        (matched_cost, matched_nodes, matched_edges) = a_star_search(
            G1, G2, cost_matrix_nodes, cost_matrix_edges, max_beam_size
        )
        return (
            matched_cost,
            get_sorted_mapping(matched_nodes, G1.num_nodes(), G2.num_nodes()),
            get_sorted_mapping(matched_edges, G1.num_edges(), G2.num_edges()),
        )

    elif algorithm == "hausdorff":
        hausdorff_cost = hausdorff_matching(
            G1,
            G2,
            node_substitution_cost,
            edge_substitution_cost,
            G1_node_deletion_cost,
            G1_edge_deletion_cost,
            G2_node_insertion_cost,
            G2_edge_insertion_cost,
        )

        return (hausdorff_cost, None, None)

    else:
        cost_matrix = contextual_cost_matrix_construction(
            G1,
            G2,
            node_substitution_cost,
            edge_substitution_cost,
            G1_node_deletion_cost,
            G1_edge_deletion_cost,
            G2_node_insertion_cost,
            G2_edge_insertion_cost,
        )
        # Match the nodes as per the LAP solution
        row_ind, col_ind, _ = lapjv(cost_matrix)

        (
            matched_cost,
            matched_nodes,
            matched_edges,
        ) = edit_cost_from_node_matching(
            G1, G2, cost_matrix_nodes, cost_matrix_edges, row_ind
        )

        return (
            matched_cost,
            get_sorted_mapping(matched_nodes, G1.num_nodes(), G2.num_nodes()),
            get_sorted_mapping(matched_edges, G1.num_edges(), G2.num_edges()),
        )
