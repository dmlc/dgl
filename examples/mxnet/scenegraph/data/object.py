"""Pascal VOC object detection dataset."""
from __future__ import absolute_import
from __future__ import division
import os
import logging
import warnings
import json
import pickle
import numpy as np
import mxnet as mx
from gluoncv.data.base import VisionDataset
from collections import Counter

class VGObject(VisionDataset):
    def __init__(self, root=os.path.join('~', '.mxnet', 'datasets', 'visualgenome'),
                 top_frequent_obj=150, split='all'):
        super(VGObject, self).__init__(root)
        self._im_shapes = {}
        self._root = os.path.expanduser(root)
        if split == 'all':
            self._dict_path = os.path.join(self._root, 'objects.json')
            self._img_path = os.path.join(self._root, 'VG_100K', '{}.jpg')
        elif split == 'train':
            self._dict_path = os.path.join(self._root, 'objects_train.json')
            self._img_path = os.path.join(self._root, 'train', '{}.jpg')
        elif split == 'val':
            self._dict_path = os.path.join(self._root, 'objects_val.json')
            self._img_path = os.path.join(self._root, 'val', '{}.jpg')
        else:
            raise NotImplementedError
        with open(self._dict_path) as f:
            tmp = f.read()
            self._ori_dict = json.loads(tmp)
        self._classes_pkl = os.path.join(self._root, 'classes.pkl')
        with open(self._classes_pkl, 'rb') as f:
            vg_obj_classes, vg_rel_classes = pickle.load(f)
        self._obj_classes = sorted(vg_obj_classes[0:top_frequent_obj])
        self._obj_classes_dict = {}
        for i, obj in enumerate(self._obj_classes):
            self._obj_classes_dict[obj] = i

        obj_alias_path = os.path.join(path, 'object_alias.txt')
        with open(obj_alias_path) as f:
            obj_alias = f.read().split('\n')
        obj_alias_dict = {}
        for obj in obj_alias:
            if len(obj) > 0:
                k, v = obj.split(',')
                obj_alias_dict[k] = v
        self.obj_alias_dict = obj_alias_dict

        _dict = []
        for it in self._ori_dict:
            label = []
            for objects in it['objects']:
                if len(objects['names']) <= 0:
                    continue
                obj_cls = objects['names'][0]
                if obj_cls in self.obj_alias_dict:
                    obj_cls = self.obj_alias_dict[obj_cls]
                if obj_cls not in self._obj_classes_dict:
                    continue
                cls = self._obj_classes_dict[obj_cls]
                xmin = objects['x']
                ymin = objects['y']
                xmax = objects['w'] + xmin
                ymax = objects['h'] + ymin
                label.append([xmin, ymin, xmax, ymax, cls])
            if len(label) <= 0:
                continue
            _dict.append({'image_id': it['image_id'],
                          'label': label})
        self._dict = _dict

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, idx):
        item = self._dict[idx]

        img_id = item['image_id']
        img_path = self._img_path.format(img_id)
        img = mx.image.imread(img_path)

        label = np.array(item['label'])

        return img, label
