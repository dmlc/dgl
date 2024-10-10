import math
from itertools import product

import dgl

import numpy as np
import pandas as pd
import torch
from dgl.data import DGLDataset


class EEGGraphDataset(DGLDataset):
    """Build graph, treat all nodes as the same type
    Parameters
    ----------
    x: edge weights of 8-node complete graph
        There are 1 x 64 edges
    y: labels (diseased/healthy)
    num_nodes: the number of nodes of the graph. In our case, it is 8.
    indices: Patient level indices. They are used to generate edge weights.

    Output
    ------
    a complete 8-node DGLGraph with node features and edge weights
    """

    def __init__(self, x, y, num_nodes, indices):
        # CAUTION - x and labels are memory-mapped, used as if they are in RAM.
        self.x = x
        self.labels = y
        self.indices = indices
        self.num_nodes = num_nodes

        # NOTE: this order decides the node index, keep consistent!
        self.ch_names = [
            "F7-F3",
            "F8-F4",
            "T7-C3",
            "T8-C4",
            "P7-P3",
            "P8-P4",
            "O1-P3",
            "O2-P4",
        ]

        # in the 10-10 system, in between the 2 10-20 electrodes in ch_names, used for calculating edge weights
        # Note: "01" is for "P03", and "02" is for "P04."
        self.ref_names = ["F5", "F6", "C5", "C6", "P5", "P6", "O1", "O2"]

        # edge indices source to target - 2 x E = 2 x 64
        # fully connected undirected graph so 8*8=64 edges
        self.node_ids = range(len(self.ch_names))
        self.edge_index = (
            torch.tensor(
                [[a, b] for a, b in product(self.node_ids, self.node_ids)],
                dtype=torch.long,
            )
            .t()
            .contiguous()
        )

        # edge attributes - E x 1
        # only the spatial distance between electrodes for now - standardize between 0 and 1
        self.distances = self.get_sensor_distances()
        a = np.array(self.distances)
        self.distances = (a - np.min(a)) / (np.max(a) - np.min(a))
        self.spec_coh_values = np.load("spec_coh_values.npy", allow_pickle=True)

    # sensor distances don't depend on window ID
    def get_sensor_distances(self):
        coords_1010 = pd.read_csv("standard_1010.tsv.txt", sep="\t")
        num_edges = self.edge_index.shape[1]
        distances = []
        for edge_idx in range(num_edges):
            sensor1_idx = self.edge_index[0, edge_idx]
            sensor2_idx = self.edge_index[1, edge_idx]
            dist = self.get_geodesic_distance(
                sensor1_idx, sensor2_idx, coords_1010
            )
            distances.append(dist)
        assert len(distances) == num_edges
        return distances

    def get_geodesic_distance(
        self, montage_sensor1_idx, montage_sensor2_idx, coords_1010
    ):
        def get_coord(ref_sensor, coord):
            return float(
                (coords_1010[coords_1010.label == ref_sensor][coord]).iloc[0]
            )

        # get the reference sensor in the 10-10 system for the current montage pair in 10-20 system
        ref_sensor1 = self.ref_names[montage_sensor1_idx]
        ref_sensor2 = self.ref_names[montage_sensor2_idx]

        x1 = get_coord(ref_sensor1, "x")
        y1 = get_coord(ref_sensor1, "y")
        z1 = get_coord(ref_sensor1, "z")

        x2 = get_coord(ref_sensor2, "x")
        y2 = get_coord(ref_sensor2, "y")
        z2 = get_coord(ref_sensor2, "z")

        # https://math.stackexchange.com/questions/1304169/distance-between-two-points-on-a-sphere
        r = 1  # since coords are on unit sphere
        # rounding is for numerical stability, domain is [-1, 1]
        dist = r * math.acos(
            round(((x1 * x2) + (y1 * y2) + (z1 * z2)) / (r**2), 2)
        )
        return dist

    # returns size of dataset = number of indices
    def __len__(self):
        return len(self.indices)

    # retrieve one sample from the dataset after applying all transforms
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # map input idx (ranging from 0 to __len__() inside self.indices)
        # to an idx in the whole dataset (inside self.x)
        # assert idx < len(self.indices)
        idx = self.indices[idx]
        node_features = self.x[idx]
        node_features = torch.from_numpy(node_features.reshape(8, 6))

        # spectral coherence between 2 montage channels!
        spec_coh_values = self.spec_coh_values[idx, :]

        # combine edge weights and spect coh values into one value/ one E x 1 tensor
        edge_weights = self.distances + spec_coh_values
        edge_weights = torch.tensor(edge_weights)  # trucated to integer

        # create 8-node complete graph
        src = [
            [0 for i in range(self.num_nodes)] for j in range(self.num_nodes)
        ]
        for i in range(len(src)):
            for j in range(len(src[i])):
                src[i][j] = i
        src = np.array(src).flatten()

        det = [
            [i for i in range(self.num_nodes)] for j in range(self.num_nodes)
        ]
        det = np.array(det).flatten()

        u, v = (torch.tensor(src), torch.tensor(det))
        g = dgl.graph((u, v))

        # add node features and edge features
        g.ndata["x"] = node_features
        g.edata["edge_weights"] = edge_weights
        return g, torch.tensor(idx), torch.tensor(self.labels[idx])
