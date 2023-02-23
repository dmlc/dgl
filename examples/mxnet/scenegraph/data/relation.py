"""Pascal VOC object detection dataset."""
from __future__ import absolute_import, division

import json
import logging
import os
import pickle
import warnings
from collections import Counter

import dgl

import mxnet as mx
import numpy as np
from gluoncv.data.base import VisionDataset
from gluoncv.data.transforms.presets.rcnn import (
    FasterRCNNDefaultTrainTransform,
    FasterRCNNDefaultValTransform,
)


class VGRelation(VisionDataset):
    def __init__(
        self,
        root=os.path.join("~", ".mxnet", "datasets", "visualgenome"),
        split="train",
    ):
        super(VGRelation, self).__init__(root)
        self._root = os.path.expanduser(root)
        self._img_path = os.path.join(self._root, "VG_100K", "{}")

        if split == "train":
            self._dict_path = os.path.join(
                self._root, "rel_annotations_train.json"
            )
        elif split == "val":
            self._dict_path = os.path.join(
                self._root, "rel_annotations_val.json"
            )
        else:
            raise NotImplementedError
        with open(self._dict_path) as f:
            tmp = f.read()
            self._dict = json.loads(tmp)

        self._predicates_path = os.path.join(self._root, "predicates.json")
        with open(self._predicates_path, "r") as f:
            tmp = f.read()
            self.rel_classes = json.loads(tmp)
        self.num_rel_classes = len(self.rel_classes) + 1

        self._objects_path = os.path.join(self._root, "objects.json")
        with open(self._objects_path, "r") as f:
            tmp = f.read()
            self.obj_classes = json.loads(tmp)
        self.num_obj_classes = len(self.obj_classes)

        if split == "val":
            self.img_transform = FasterRCNNDefaultValTransform(
                short=600, max_size=1000
            )
        else:
            self.img_transform = FasterRCNNDefaultTrainTransform(
                short=600, max_size=1000
            )
        self.split = split

    def __len__(self):
        return len(self._dict)

    def _hash_bbox(self, object):
        num_list = [object["category"]] + object["bbox"]
        return "_".join([str(num) for num in num_list])

    def __getitem__(self, idx):
        img_id = list(self._dict)[idx]
        img_path = self._img_path.format(img_id)
        img = mx.image.imread(img_path)

        item = self._dict[img_id]
        n_edges = len(item)

        # edge to node ids
        sub_node_hash = []
        ob_node_hash = []
        for i, it in enumerate(item):
            sub_node_hash.append(self._hash_bbox(it["subject"]))
            ob_node_hash.append(self._hash_bbox(it["object"]))
        node_set = sorted(list(set(sub_node_hash + ob_node_hash)))
        n_nodes = len(node_set)
        node_to_id = {}
        for i, node in enumerate(node_set):
            node_to_id[node] = i
        sub_id = []
        ob_id = []
        for i in range(n_edges):
            sub_id.append(node_to_id[sub_node_hash[i]])
            ob_id.append(node_to_id[ob_node_hash[i]])

        # node features
        bbox = mx.nd.zeros((n_nodes, 4))
        node_class_ids = mx.nd.zeros((n_nodes, 1))
        node_visited = [False for i in range(n_nodes)]
        for i, it in enumerate(item):
            if not node_visited[sub_id[i]]:
                ind = sub_id[i]
                sub = it["subject"]
                node_class_ids[ind] = sub["category"]
                # y1y2x1x2 to x1y1x2y2
                bbox[ind, 0] = sub["bbox"][2]
                bbox[ind, 1] = sub["bbox"][0]
                bbox[ind, 2] = sub["bbox"][3]
                bbox[ind, 3] = sub["bbox"][1]

                node_visited[ind] = True

            if not node_visited[ob_id[i]]:
                ind = ob_id[i]
                ob = it["object"]
                node_class_ids[ind] = ob["category"]
                # y1y2x1x2 to x1y1x2y2
                bbox[ind, 0] = ob["bbox"][2]
                bbox[ind, 1] = ob["bbox"][0]
                bbox[ind, 2] = ob["bbox"][3]
                bbox[ind, 3] = ob["bbox"][1]

                node_visited[ind] = True

        eta = 0.1
        node_class_vec = node_class_ids[:, 0].one_hot(
            self.num_obj_classes,
            on_value=1 - eta + eta / self.num_obj_classes,
            off_value=eta / self.num_obj_classes,
        )

        # augmentation
        if self.split == "val":
            img, bbox, _ = self.img_transform(img, bbox)
        else:
            img, bbox = self.img_transform(img, bbox)

        # build the graph
        g = dgl.DGLGraph()
        g.add_nodes(n_nodes)
        adjmat = np.zeros((n_nodes, n_nodes))
        predicate = []
        for i, it in enumerate(item):
            adjmat[sub_id[i], ob_id[i]] = 1
            predicate.append(it["predicate"])
        predicate = mx.nd.array(predicate).expand_dims(1)
        g.add_edges(sub_id, ob_id, {"rel_class": mx.nd.array(predicate) + 1})
        empty_edge_list = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j and adjmat[i, j] == 0:
                    empty_edge_list.append((i, j))
        if len(empty_edge_list) > 0:
            src, dst = tuple(zip(*empty_edge_list))
            g.add_edges(
                src, dst, {"rel_class": mx.nd.zeros((len(empty_edge_list), 1))}
            )

        # assign features
        g.ndata["bbox"] = bbox
        g.ndata["node_class"] = node_class_ids
        g.ndata["node_class_vec"] = node_class_vec

        return g, img
