import os
import sys
import numpy as np
from copy import deepcopy

import torch

from pytorch3d.structures import Meshes
from pytorch3d.io.obj_io import load_obj as load_3d_obj


def preprocess(obj_file):
    verts, faces, aux = load_3d_obj(obj_file)
    # Quantilization
    quant_verts = np.clip((verts.data.numpy()+0.5) * 31, 0, 31).astype(np.int64).astype(np.float32)
    # Sorting, first Y(up), then X(front), then Z(right)
    vert_val = quant_verts[:,1]*256*256 + quant_verts[:,0]*256 + quant_verts[:,2]

    # Unique on vertex
    uni_val, uni_val_idx = np.unique(vert_val, return_index=True)
    _, uni_val_inverse_idx = np.unique(vert_val, return_inverse=True)
    uni_new_verts = quant_verts[uni_val_idx, :]

    # Unique on faces
    face_verts_idx = deepcopy(faces[0].data.numpy())
    face_verts_idx[:,0] = uni_val_inverse_idx[face_verts_idx[:, 0]]
    face_verts_idx[:,1] = uni_val_inverse_idx[face_verts_idx[:, 1]]
    face_verts_idx[:,2] = uni_val_inverse_idx[face_verts_idx[:, 2]]
    uni_face_verts_idx = np.unique(face_verts_idx, axis=0)
    # If in one face, two vertexs are the same, remove that face
    same_vert_0_1 = uni_face_verts_idx[:,0] == uni_face_verts_idx[:,1]
    same_vert_0_2 = uni_face_verts_idx[:,0] == uni_face_verts_idx[:,2]
    same_vert_1_2 = uni_face_verts_idx[:,1] == uni_face_verts_idx[:,2]
    uni_vert_within_face = np.logical_not(np.any([same_vert_0_1, same_vert_0_2, same_vert_1_2], axis=0))
    uni_face_verts_idx_within_face = uni_face_verts_idx[uni_vert_within_face]

    # Sort vertex
    sort_val = uni_new_verts[:,1]*256*256 + uni_new_verts[:,0]*256 + uni_new_verts[:,2]
    sort_idx = np.argsort(sort_val)
    processed_verts = uni_new_verts[sort_idx]
    processed_face_verts_idx = deepcopy(uni_face_verts_idx_within_face)
    processed_face_verts_idx[:,0] = sort_idx[processed_face_verts_idx[:,0]]
    processed_face_verts_idx[:,1] = sort_idx[processed_face_verts_idx[:,1]]
    processed_face_verts_idx[:,2] = sort_idx[processed_face_verts_idx[:,2]]

    # Also sort vertex idx within face
    # TODO: this seems changing the rendering result?
    processed_face_verts_idx = np.sort(processed_face_verts_idx, axis=1)
    # Sort faces by vertex index
    num_verts = processed_verts.shape[0]
    face_sort_val = processed_face_verts_idx[:,0]*num_verts*num_verts+\
                    processed_face_verts_idx[:,1]*num_verts+\
                    processed_face_verts_idx[:,2]
    sort_face_idx = np.argsort(face_sort_val)
    processed_face_verts_idx = processed_face_verts_idx[sort_face_idx, :]
    return processed_verts, processed_face_verts_idx


if __name__ == '__main__':
    # in short experiment, we only train chair and table
    class_ids = ['04379243', '03001627']
    dataset_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
    list_file_name = dataset_dir + 'table_chair.txt'
    with open(list_file_name, 'w') as lf:
        for class_id in class_ids:
            class_dir = os.path.join(dataset_dir, class_id)
            subdirs = [os.path.join(class_dir, f) for f in os.listdir(class_dir) if os.path.isdir(os.path.join(class_dir, f))]
            for each_dir in subdirs:
                obj_file = os.path.join(each_dir, "models/model_normalized.obj")
                verts, faces = preprocess(obj_file)
                if verts.shape[0] < 800 and faces.shape[0] < 2800:
                    print (obj_file)
                    lf.write(obj_file + '\n')
