"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import json
import dgl
import pickle
import numpy as np
import mxnet as mx
from gluoncv.data.base import VisionDataset
from collections import Counter
from gluoncv.data.transforms.presets.rcnn import FasterRCNNDefaultTrainTransform, FasterRCNNDefaultValTransform

def get_name(obj):
    if 'name' in obj:
        return obj['name']
    elif 'names' in obj:
        return obj['names'][0]
    else:
        return ''

class VGRelation(VisionDataset):
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'visualgenome'),
                 top_frequent_rel=50, top_frequent_obj=150,
                 split='all', balancing='sample', rel_json_path=None):
        super(VGRelation, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        if split == 'all':
            self._dict_path = os.path.join(self._root, 'relationships.json')
            self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
        elif split == 'train':
            self._dict_path = os.path.join(self._root, 'relationships_train.json')
            self._img_path = os.path.join(self._root, 'train', '{}.jpg')
        elif split == 'val':
            self._dict_path = os.path.join(self._root, 'relationships_val.json')
            self._img_path = os.path.join(self._root, 'val', '{}.jpg')
        elif split == 'custom':
            if rel_json_path is not None:
                self._dict_path = os.path.join(self._root, rel_json_path)
                self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
            else:
                raise ValueError("Must set value for rel_json_path when split=='custom'.")
        else:
            raise NotImplementedError
        with open(self._dict_path) as f:
            tmp = f.read()
            self._dict = json.loads(tmp)
        self._classes_pkl = os.path.join(self._root, 'classes.pkl')
        with open(self._classes_pkl, 'rb') as f:
            vg_obj_classes, vg_rel_classes = pickle.load(f)
        self._obj_classes = sorted(vg_obj_classes[0:top_frequent_obj])
        self._relations = sorted(vg_rel_classes[0:top_frequent_rel])
        self._relations_dict = {}
        for i, rel in enumerate(self._relations):
            self._relations_dict[rel] = i

        self._obj_classes_dict = {}
        for i, obj in enumerate(self._obj_classes):
            self._obj_classes_dict[obj] = i

        self._balancing = balancing
        if split == 'val':
            self.img_transform = FasterRCNNDefaultValTransform(short=600, max_size=1000)
        else:
            self.img_transform = FasterRCNNDefaultTrainTransform(short=600, max_size=1000)
        self.split = split

        obj_alias_path = os.path.join(self._root, 'object_alias.txt')
        rel_alias_path = os.path.join(self._root, 'relationship_alias.txt')
        with open(obj_alias_path) as f:
            obj_alias = f.read().split('\n')
        obj_alias_dict = {}
        for obj in obj_alias:
            if len(obj) > 0:
                tmp = obj.split(',')
                k = tmp[0]
                v = tmp[1]
                obj_alias_dict[k] = v
        self.obj_alias_dict = obj_alias_dict

        with open(rel_alias_path) as f:
            rel_alias = f.read().split('\n')
        rel_alias_dict = {}
        for rel in rel_alias:
            if len(rel) > 0:
                tmp = rel.split(',')
                k = tmp[0]
                v = tmp[1]
                rel_alias_dict[k] = v
        self.rel_alias_dict = rel_alias_dict

    def __len__(self):
        return len(self._dict)

    def _extract_label(self, rel):
        n = len(rel)
        # extract global ids first, and map into local ids
        # object_ids = [0]
        object_ids = []
        keep_inds = []
        for i, rl in enumerate(rel):
            sub = rl['subject']
            ob = rl['object']
            if sub['object_id'] == ob['object_id']:
                continue

            k = get_name(sub)
            if len(k) == 0:
                continue
            if k in self.obj_alias_dict:
                k = self.obj_alias_dict[k]
            if not k in self._obj_classes_dict:
                continue

            k = get_name(ob)
            if len(k) == 0:
                continue
            if k in self.obj_alias_dict:
                k = self.obj_alias_dict[k]
            if k not in self._obj_classes_dict:
                continue

            if len(rl['predicate']) == 0:
                continue
            else:
                synset = rl['predicate']
                if synset in self.rel_alias_dict:
                    synset = self.rel_alias_dict[synset]
                if synset not in self._relations_dict:
                    continue

            object_ids.append(sub['object_id'])
            object_ids.append(ob['object_id'])
            keep_inds.append(i)
        object_ids = list(set(object_ids))

        ids_dict = {}
        for i, obj in enumerate(object_ids):
            ids_dict[str(obj)] = i
        m = len(object_ids)
        if m == 0:
            return None, None, None
        bbox = mx.nd.zeros((m, 4))
        node_class = mx.nd.zeros((m))
        visit_ind = set()
        edges = {'src': [],
                 'dst': [],
                 'rel': [],
                 'link': []}
        keep_inds = set(keep_inds)
        for i, rl in enumerate(rel):
            if i not in keep_inds:
                continue
            # extract xyhw and remap object id
            sub = rl['subject']
            ob = rl['object']
            sub_key = str(sub['object_id'])
            ob_key = str(ob['object_id'])
            if sub_key not in ids_dict or ob_key not in ids_dict:
                continue
            sub_ind = ids_dict[sub_key]
            ob_ind = ids_dict[ob_key]

            if sub_ind not in visit_ind:
                visit_ind.add(sub_ind)
                bbox[sub_ind,] = mx.nd.array([sub['x'], sub['y'],
                                              sub['w'] + sub['x'], sub['h'] + sub['y']])
                k = get_name(sub)
                if k in self.obj_alias_dict:
                    k = self.obj_alias_dict[k]
                node_class[sub_ind] = self._obj_classes_dict[k]

            if ob_ind not in visit_ind:
                visit_ind.add(ob_ind)
                bbox[ob_ind,] = mx.nd.array([ob['x'], ob['y'],
                                             ob['w'] + ob['x'], ob['h'] + ob['y']])
                k = get_name(ob)
                if k in self.obj_alias_dict:
                    k = self.obj_alias_dict[k]
                node_class[ob_ind] = self._obj_classes_dict[k]

            # relational label id of the edge
            synset = rl['predicate']
            if synset in self.rel_alias_dict:
                synset = self.rel_alias_dict[synset]
            rel_idx = self._relations_dict[synset]

            edges['src'].append(sub_ind)
            edges['dst'].append(ob_ind)
            edges['rel'].append(rel_idx)
        return edges, bbox, node_class.expand_dims(1)

    def _build_complete_graph(self, edges, bbox, node_class, img, img_id):
        N = bbox.shape[0]
        g = dgl.DGLGraph()
        g.add_nodes(N)

        # complete graph
        edge_list = []
        for i in range(N-1):
            for j in range(i+1, N):
                edge_list.append((i, j))
        src, dst = tuple(zip(*edge_list))
        g.add_edges(src, dst)
        g.add_edges(dst, src)

        # node features
        g.ndata['bbox'] = bbox

        g.ndata['node_class_ids'] = node_class
        eta = 0.1
        n_classes = len(self._obj_classes_dict)
        g.ndata['node_class_vec'] = node_class[:,0].one_hot(n_classes,
                                                            on_value = 1 - eta + eta/n_classes,
                                                            off_value = eta / n_classes)

        # assign class label to edges
        eids = g.edge_ids(edges['src'], edges['dst'])
        n = g.number_of_edges()
        k = eids.shape[0]

        classes = np.zeros((n))
        classes[eids.asnumpy()] = edges['rel']
        g.edata['classes'] = mx.nd.array(classes)
        links = np.zeros((n))
        links[eids.asnumpy()] = 1
        if links.sum() == 0:
            return None
        g.edata['link'] = mx.nd.array(links)

        if self._balancing == 'weight':
            if n == k:
                w0 = 0
            else:
                w0 = 1.0 / (2 * (n - k))
            if k == 0:
                wn = 0
            else:
                wn = 1.0 / (2 * k)
            weights = np.zeros((n)) + w0
            weights[eids.asnumpy()] = wn
        elif self._balancing == 'sample':
            sample_ind = np.random.randint(0, n, 2*k)
            weights = np.zeros((n))
            weights[sample_ind] = 1
            weights[eids.asnumpy()] = 1
        else:
            raise NotImplementedError
        g.edata['weights'] = mx.nd.array(weights)

        return g

    def __getitem__(self, idx):
        item = self._dict[idx]

        img_id = item['image_id']
        rel = item['relationships']

        img_path = self._img_path.format(img_id)
        img = mx.image.imread(img_path)

        edges, bbox, node_class = self._extract_label(rel)
        if edges is None:
            return None, None
        if self.split == 'val':
            img, bbox, _ = self.img_transform(img, bbox)
        else:
            img, bbox = self.img_transform(img, bbox)
        if bbox.shape[0] < 2:
            return None, None
        g = self._build_complete_graph(edges, bbox, node_class, img, img_id)

        return g, img
