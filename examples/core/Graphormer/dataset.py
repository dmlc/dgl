"""
This file contains the MolHIVDataset class, which handles data preprocessing
(computing required graph features, converting graphs to tensors) of the
ogbg-molhiv dataset.
"""
import torch as th
import torch.nn.functional as F
from dgl import shortest_dist
from ogb.graphproppred import DglGraphPropPredDataset
from torch.nn.utils.rnn import pad_sequence


class MolHIVDataset(th.utils.data.Dataset):
    def __init__(self):
        dataset = DglGraphPropPredDataset(name="ogbg-molhiv")
        split_idx = dataset.get_idx_split()

        # Compute the shortest path distances and their corresponding paths
        # of all graphs during preprocessing.
        for g, label in dataset:
            spd, path = shortest_dist(g, root=None, return_paths=True)
            g.ndata["spd"] = spd
            g.ndata["path"] = path

        self.train, self.val, self.test = (
            dataset[split_idx["train"]],
            dataset[split_idx["valid"]],
            dataset[split_idx["test"]],
        )

    def collate(self, samples):
        # To match Graphormer's input style, all graph features should be
        # padded to the same size. Keep in mind that different graphs may
        # have varying feature sizes since they have different number of
        # nodes, so they will be aligned with the graph having the maximum
        # number of nodes.
        graphs, labels = map(list, zip(*samples))
        labels = th.stack(labels)

        num_graphs = len(graphs)
        num_nodes = [g.num_nodes() for g in graphs]
        max_num_nodes = max(num_nodes)

        # Graphormer adds a virual node to the graph, which is connected to
        # all other nodes and supposed to represent the graph embedding. So
        # here +1 is for the virtual node.
        attn_mask = th.zeros(num_graphs, max_num_nodes + 1, max_num_nodes + 1)
        node_feat = []
        in_degree, out_degree = [], []
        path_data = []
        # Since shortest_dist returns -1 for unreachable node pairs and padded
        # nodes are unreachable to others, distance relevant to padded nodes
        # use -1 padding as well.
        dist = -th.ones(
            (num_graphs, max_num_nodes, max_num_nodes), dtype=th.long
        )

        for i in range(num_graphs):
            # A binary mask where invalid positions are indicated by True.
            attn_mask[i, :, num_nodes[i] + 1 :] = 1

            # +1 to distinguish padded non-existing nodes from real nodes
            node_feat.append(graphs[i].ndata["feat"] + 1)

            in_degree.append(
                th.clamp(graphs[i].in_degrees() + 1, min=0, max=512)
            )
            out_degree.append(
                th.clamp(graphs[i].out_degrees() + 1, min=0, max=512)
            )

            # Path padding to make all paths to the same length "max_len".
            path = graphs[i].ndata["path"]
            path_len = path.size(dim=2)
            # shape of shortest_path: [n, n, max_len]
            max_len = 5
            if path_len >= max_len:
                shortest_path = path[:, :, :max_len]
            else:
                p1d = (0, max_len - path_len)
                # Use the same -1 padding as shortest_dist for
                # invalid edge IDs.
                shortest_path = F.pad(path, p1d, "constant", -1)
            pad_num_nodes = max_num_nodes - num_nodes[i]
            p3d = (0, 0, 0, pad_num_nodes, 0, pad_num_nodes)
            shortest_path = F.pad(shortest_path, p3d, "constant", -1)
            # +1 to distinguish padded non-existing edges from real edges
            edata = graphs[i].edata["feat"] + 1
            # shortest_dist pads non-existing edges (at the end of shortest
            # paths) with edge IDs -1, and th.zeros(1, edata.shape[1]) stands
            # for all padded edge features.
            edata = th.cat(
                (edata, th.zeros(1, edata.shape[1]).to(edata.device)), dim=0
            )
            path_data.append(edata[shortest_path])

            dist[i, : num_nodes[i], : num_nodes[i]] = graphs[i].ndata["spd"]

        # node feat padding
        node_feat = pad_sequence(node_feat, batch_first=True)

        # degree padding
        in_degree = pad_sequence(in_degree, batch_first=True)
        out_degree = pad_sequence(out_degree, batch_first=True)

        return (
            labels.reshape(num_graphs, -1),
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            th.stack(path_data),
            dist,
        )
