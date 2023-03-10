import os

import dgl

import numpy
import pandas
import torch


def process_raw_data(raw_dir, processed_dir):
    r"""

    Description
    -----------
    Preprocess Elliptic dataset like the EvolveGCN official instruction:
    github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
    The main purpose is to convert original idx to contiguous idx start at 0.
    """
    oid_nid_path = os.path.join(processed_dir, "oid_nid.npy")
    id_label_path = os.path.join(processed_dir, "id_label.npy")
    id_time_features_path = os.path.join(processed_dir, "id_time_features.npy")
    src_dst_time_path = os.path.join(processed_dir, "src_dst_time.npy")
    if (
        os.path.exists(oid_nid_path)
        and os.path.exists(id_label_path)
        and os.path.exists(id_time_features_path)
        and os.path.exists(src_dst_time_path)
    ):
        print(
            "The preprocessed data already exists, skip the preprocess stage!"
        )
        return
    print("starting process raw data in {}".format(raw_dir))
    id_label = pandas.read_csv(
        os.path.join(raw_dir, "elliptic_txs_classes.csv")
    )
    src_dst = pandas.read_csv(
        os.path.join(raw_dir, "elliptic_txs_edgelist.csv")
    )
    # elliptic_txs_features.csv has no header, and it has the same order idx with elliptic_txs_classes.csv
    id_time_features = pandas.read_csv(
        os.path.join(raw_dir, "elliptic_txs_features.csv"), header=None
    )

    # get oldId_newId
    oid_nid = id_label.loc[:, ["txId"]]
    oid_nid = oid_nid.rename(columns={"txId": "originalId"})
    oid_nid.insert(1, "newId", range(len(oid_nid)))

    # map classes unknown,1,2 to -1,1,0 and construct id_label. type 1 means illicit.
    id_label = pandas.concat(
        [
            oid_nid["newId"],
            id_label["class"].map({"unknown": -1.0, "1": 1.0, "2": 0.0}),
        ],
        axis=1,
    )

    # replace originalId to newId.
    # Attention: the timestamp in features start at 1.
    id_time_features[0] = oid_nid["newId"]

    # construct originalId2newId dict
    oid_nid_dict = oid_nid.set_index(["originalId"])["newId"].to_dict()
    # construct newId2timestamp dict
    nid_time_dict = id_time_features.set_index([0])[1].to_dict()

    # Map id in edgelist to newId, and add a timestamp to each edge.
    # Attention: From the EvolveGCN official instruction, the timestamp with edgelist start at 0, rather than 1.
    # see: github.com/IBM/EvolveGCN/blob/master/elliptic_construction.md
    # Here we dose not follow the official instruction, which means timestamp with edgelist also start at 1.
    # In EvolveGCN example, the edge timestamp will not be used.
    #
    # Note: in the dataset, src and dst node has the same timestamp, so it's easy to set edge's timestamp.
    new_src = src_dst["txId1"].map(oid_nid_dict).rename("newSrc")
    new_dst = src_dst["txId2"].map(oid_nid_dict).rename("newDst")
    edge_time = new_src.map(nid_time_dict).rename("timestamp")
    src_dst_time = pandas.concat([new_src, new_dst, edge_time], axis=1)

    # save oid_nid, id_label, id_time_features, src_dst_time to disk. we can convert them to numpy.
    # oid_nid: type int.  id_label: type int.  id_time_features: type float.  src_dst_time: type int.
    oid_nid = oid_nid.to_numpy(dtype=int)
    id_label = id_label.to_numpy(dtype=int)
    id_time_features = id_time_features.to_numpy(dtype=float)
    src_dst_time = src_dst_time.to_numpy(dtype=int)

    numpy.save(oid_nid_path, oid_nid)
    numpy.save(id_label_path, id_label)
    numpy.save(id_time_features_path, id_time_features)
    numpy.save(src_dst_time_path, src_dst_time)
    print(
        "Process Elliptic raw data done, data has saved into {}".format(
            processed_dir
        )
    )


class EllipticDataset:
    def __init__(
        self, raw_dir, processed_dir, self_loop=True, reverse_edge=True
    ):
        self.raw_dir = raw_dir
        self.processd_dir = processed_dir
        self.self_loop = self_loop
        self.reverse_edge = reverse_edge

    def process(self):
        process_raw_data(self.raw_dir, self.processd_dir)
        id_time_features = torch.Tensor(
            numpy.load(os.path.join(self.processd_dir, "id_time_features.npy"))
        )
        id_label = torch.IntTensor(
            numpy.load(os.path.join(self.processd_dir, "id_label.npy"))
        )
        src_dst_time = torch.IntTensor(
            numpy.load(os.path.join(self.processd_dir, "src_dst_time.npy"))
        )

        src = src_dst_time[:, 0]
        dst = src_dst_time[:, 1]
        # id_label[:, 0] is used to add self loop
        if self.self_loop:
            if self.reverse_edge:
                g = dgl.graph(
                    data=(
                        torch.cat((src, dst, id_label[:, 0])),
                        torch.cat((dst, src, id_label[:, 0])),
                    ),
                    num_nodes=id_label.shape[0],
                )
                g.edata["timestamp"] = torch.cat(
                    (
                        src_dst_time[:, 2],
                        src_dst_time[:, 2],
                        id_time_features[:, 1].int(),
                    )
                )
            else:
                g = dgl.graph(
                    data=(
                        torch.cat((src, id_label[:, 0])),
                        torch.cat((dst, id_label[:, 0])),
                    ),
                    num_nodes=id_label.shape[0],
                )
                g.edata["timestamp"] = torch.cat(
                    (src_dst_time[:, 2], id_time_features[:, 1].int())
                )
        else:
            if self.reverse_edge:
                g = dgl.graph(
                    data=(torch.cat((src, dst)), torch.cat((dst, src))),
                    num_nodes=id_label.shape[0],
                )
                g.edata["timestamp"] = torch.cat(
                    (src_dst_time[:, 2], src_dst_time[:, 2])
                )
            else:
                g = dgl.graph(data=(src, dst), num_nodes=id_label.shape[0])
                g.edata["timestamp"] = src_dst_time[:, 2]

        time_features = id_time_features[:, 1:]
        label = id_label[:, 1]
        g.ndata["label"] = label
        g.ndata["feat"] = time_features

        # used to construct time-based sub-graph.
        node_mask_by_time = []
        start_time = int(torch.min(id_time_features[:, 1]))
        end_time = int(torch.max(id_time_features[:, 1]))
        for i in range(start_time, end_time + 1):
            node_mask = id_time_features[:, 1] == i
            node_mask_by_time.append(node_mask)

        return g, node_mask_by_time

    @property
    def num_classes(self):
        r"""Number of classes for each node."""
        return 2
