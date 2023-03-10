import os

import graph_tool as gt
import graph_tool.topology as gt_topology
import networkx as nx
import numpy as np
import torch

from dgl.data.utils import load_graphs, save_graphs
from ogb.graphproppred import DglGraphPropPredDataset
from tqdm import tqdm


def to_undirected(edge_index):
    row, col = edge_index.transpose(1, 0)
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    return edge_index.transpose(1, 0).tolist()


def induced_edge_automorphism_orbits(edge_list):
    ##### node automorphism orbits #####
    graph = gt.Graph(directed=False)
    graph.add_edge_list(edge_list)
    gt.stats.remove_self_loops(graph)
    gt.stats.remove_parallel_edges(graph)

    # compute the node automorphism group
    aut_group = gt_topology.subgraph_isomorphism(
        graph, graph, induced=False, subgraph=True, generator=False
    )

    orbit_membership = {}
    for v in graph.get_vertices():
        orbit_membership[v] = v

    # whenever two nodes can be mapped via some automorphism, they are assigned the same orbit
    for aut in aut_group:
        for original, node in enumerate(aut):
            role = min(original, orbit_membership[node])
            orbit_membership[node] = role

    orbit_membership_list = [[], []]
    for node, om_curr in orbit_membership.items():
        orbit_membership_list[0].append(node)
        orbit_membership_list[1].append(om_curr)

    # make orbit list contiguous (i.e. 0,1,2,...O)
    _, contiguous_orbit_membership = np.unique(
        orbit_membership_list[1], return_inverse=True
    )

    orbit_membership = {
        node: contiguous_orbit_membership[i]
        for i, node in enumerate(orbit_membership_list[0])
    }

    aut_count = len(aut_group)

    ##### induced edge automorphism orbits (according to the node automorphism group) #####
    edge_orbit_partition = dict()
    edge_orbit_membership = dict()
    edge_orbits2inds = dict()
    ind = 0

    edge_list = to_undirected(torch.tensor(graph.get_edges()))

    # infer edge automorphisms from the node automorphisms
    for i, edge in enumerate(edge_list):
        edge_orbit = frozenset(
            [orbit_membership[edge[0]], orbit_membership[edge[1]]]
        )
        if edge_orbit not in edge_orbits2inds:
            edge_orbits2inds[edge_orbit] = ind
            ind_edge_orbit = ind
            ind += 1
        else:
            ind_edge_orbit = edge_orbits2inds[edge_orbit]

        if ind_edge_orbit not in edge_orbit_partition:
            edge_orbit_partition[ind_edge_orbit] = [tuple(edge)]
        else:
            edge_orbit_partition[ind_edge_orbit] += [tuple(edge)]

        edge_orbit_membership[i] = ind_edge_orbit

    print(
        "Edge orbit partition of given substructure: {}".format(
            edge_orbit_partition
        )
    )
    print("Number of edge orbits: {}".format(len(edge_orbit_partition)))
    print("Graph (node) automorphism count: {}".format(aut_count))

    return graph, edge_orbit_partition, edge_orbit_membership, aut_count


def subgraph_isomorphism_edge_counts(edge_index, subgraph_dict):
    ##### edge structural identifiers #####

    edge_index = edge_index.transpose(1, 0).cpu().numpy()
    edge_dict = {}
    for i, edge in enumerate(edge_index):
        edge_dict[tuple(edge)] = i

    subgraph_edges = to_undirected(
        torch.tensor(subgraph_dict["subgraph"].get_edges().tolist())
    )

    G_gt = gt.Graph(directed=False)
    G_gt.add_edge_list(list(edge_index))
    gt.stats.remove_self_loops(G_gt)
    gt.stats.remove_parallel_edges(G_gt)

    # compute all subgraph isomorphisms
    sub_iso = gt_topology.subgraph_isomorphism(
        subgraph_dict["subgraph"],
        G_gt,
        induced=True,
        subgraph=True,
        generator=True,
    )

    counts = np.zeros(
        (edge_index.shape[0], len(subgraph_dict["orbit_partition"]))
    )

    for sub_iso_curr in sub_iso:
        mapping = sub_iso_curr.get_array()
        for i, edge in enumerate(subgraph_edges):
            # for every edge in the graph H, find the edge in the subgraph G_S to which it is mapped
            # (by finding where its endpoints are matched).
            # Then, increase the count of the matched edge w.r.t. the corresponding orbit
            # Repeat for the reverse edge (the one with the opposite direction)

            edge_orbit = subgraph_dict["orbit_membership"][i]
            mapped_edge = tuple([mapping[edge[0]], mapping[edge[1]]])
            counts[edge_dict[mapped_edge], edge_orbit] += 1

    counts = counts / subgraph_dict["aut_count"]

    counts = torch.tensor(counts)

    return counts


def prepare_dataset(name):
    # maximum size of cycle graph
    k = 8

    path = os.path.join("./", "dataset", name)
    data_folder = os.path.join(path, "processed")
    os.makedirs(data_folder, exist_ok=True)

    data_file = os.path.join(
        data_folder, "cycle_graph_induced_{}.bin".format(k)
    )

    # try to load
    if os.path.exists(data_file):  # load
        print("Loading dataset from {}".format(data_file))
        g_list, split_idx = load_graphs(data_file)
    else:  # generate
        g_list, split_idx = generate_dataset(path, name)
        print("Saving dataset to {}".format(data_file))
        save_graphs(data_file, g_list, split_idx)

    return g_list, split_idx


def generate_dataset(path, name):
    ### compute the orbits of each substructure in the list, as well as the node automorphism count
    subgraph_dicts = []

    edge_lists = []
    for k in range(3, 8 + 1):
        graphs_nx = nx.cycle_graph(k)
        edge_lists.append(list(graphs_nx.edges))

    for edge_list in edge_lists:
        (
            subgraph,
            orbit_partition,
            orbit_membership,
            aut_count,
        ) = induced_edge_automorphism_orbits(edge_list=edge_list)
        subgraph_dicts.append(
            {
                "subgraph": subgraph,
                "orbit_partition": orbit_partition,
                "orbit_membership": orbit_membership,
                "aut_count": aut_count,
            }
        )

    ### load and preprocess dataset
    dataset = DglGraphPropPredDataset(name=name, root=path)
    split_idx = dataset.get_idx_split()

    # computation of subgraph isomorphisms & creation of data structure
    graphs_dgl = list()
    split_idx["label"] = []
    for i, datapoint in tqdm(enumerate(dataset)):
        g, label = datapoint
        g = _prepare(g, subgraph_dicts)
        graphs_dgl.append(g)
        split_idx["label"].append(label)

    split_idx["label"] = torch.stack(split_idx["label"])

    return graphs_dgl, split_idx


def _prepare(g, subgraph_dicts):
    edge_index = torch.stack(g.edges())

    identifiers = None
    for subgraph_dict in subgraph_dicts:
        counts = subgraph_isomorphism_edge_counts(edge_index, subgraph_dict)
        identifiers = (
            counts
            if identifiers is None
            else torch.cat((identifiers, counts), 1)
        )

    g.edata["subgraph_counts"] = identifiers.long()

    return g


if __name__ == "__main__":
    prepare_dataset("ogbg-molpcba")
