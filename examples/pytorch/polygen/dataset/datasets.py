from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from .vertexgraph import *
from .facegraph import *
from .preprocess_mesh import preprocess as preprocess_mesh_obj

VertexDataTuple = namedtuple('VertexDataTuple', ['g', 'num_edges_dd', 'tgt', 'mode', 'device'])
VertexGraph = namedtuple('VertexGraph',
                   ['g', 'tgt', 'tgt_y', 'nids', 'eids',  'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])
FaceDataTuple = namedtuple('FaceDataTuple', ['g', 'num_edges_enc', 'num_edges_ed', 'num_edges_dd', 'src', 'tgt', 'mode', 'device'])
FaceGraph = namedtuple('FaceGraph',
                   ['g', 'src', 'tgt', 'tgt_y', 'nids', 'eids', 'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])



class VertexDataset(Dataset):
    """Vertex Dataset"""
    COORD_BIN = 256
    INIT_BIN = COORD_BIN
    EOS_BIN = COORD_BIN + 1
    PAD_BIN = COORD_BIN + 2
    MAX_VERT_LENGTH = 800
    MAX_LENGTH = MAX_VERT_LENGTH * 3 + 2
    
    def __init__(self, dataset_list_path, mode):
        dataset_list_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
        dataset_list_path = os.path.join(dataset_list_dir, dataset_list_path)
        self.mode = mode
        with open(dataset_list_path, 'r') as f:
            self.dataset_list = f.readlines()
        self.pad_id = self.PAD_BIN
        self.graphpool = VertexNetGraphPool(n=self.MAX_LENGTH)
 
    @property
    def vocab_size(self):
        return self.COORD_BIN+3 

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        obj_file = self.dataset_list[idx].strip()
        try:
            verts, faces = preprocess_mesh_obj(obj_file, self.COORD_BIN)
        except:
            # If fail, try another random example
            to_try = np.random.randint(0, len(self.dataset_list))
            return self.__getitem__(to_try)
        # Flattern verts, order Y(up), X(front), Z(right)
        reordered_verts = np.zeros_like(verts)
        reordered_verts[:,0] = verts[:,1]
        reordered_verts[:,1] = verts[:,0]
        reordered_verts[:,2] = verts[:,2]
        flattern_verts = [self.INIT_BIN] + reordered_verts.flatten().astype(np.int64).tolist() + [self.EOS_BIN]
        # exp
        if len(flattern_verts) > self.MAX_LENGTH:
            to_try = np.random.randint(0, len(self.dataset_list))
            return self.__getitem__(to_try)
        return VertexDataTuple(g=self.graphpool.g_pool.get_graph_for_size(len(flattern_verts)-1),
                num_edges_dd=self.graphpool.num_edges['dd'][len(flattern_verts)-1],
                tgt=flattern_verts, mode=self.mode, device='cpu')

# different collate function for infer and train?
def collate_vertexgraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 3-tuples.
       Each tuples contains a graph, num_edges, tgt sequence, mode, device 
    Returns
    -------
    g : tuple of batched DGLGraph and input tensors
    """
    print ('hahaha')
    assert len(data[0]) == 5, \
            'Expect the tuple to be of length 5, got {:d}'.format(len(data[0]))
    g_list, num_edges_dd, tgt_buf, modes, devices = map(list, zip(*data))
    num_edges = {'dd': num_edges_dd}
    tgt_lens = [len(_) - 1 for _ in tgt_buf]
    mode = modes[0]
    device = devices[0]
    g = dgl.batch(g_list)
    tgt, tgt_y = [], []
    tgt_pos = []
    dec_ids = []
    d2d_eids = []
    n_nodes, n_edges, n_tokens = 0, 0, 0
    for tgt_sample, n, n_dd in zip(tgt_buf, tgt_lens, num_edges['dd']):
        tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
        tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
        tgt_pos.append(th.arange(n, dtype=th.long, device=device))
        dec_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
        d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
        n_nodes += n
        n_tokens += n
        n_edges += n_dd

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    return VertexGraph(g=g,
                 tgt=(th.cat(tgt), th.cat(tgt_pos)),
                 tgt_y=th.cat(tgt_y),
                 nids = {'dec': th.cat(dec_ids)},
                 eids = {'dd': th.cat(d2d_eids)},
                 nid_arr = {'dec': dec_ids},
                 n_nodes=n_nodes,
                 n_edges=n_edges,
                 n_tokens=n_tokens)


class FaceDataset(object):
    '''
    Dataset class for ShapeNet Vertex.
    '''
    COORD_BIN = 256
    INIT_BIN = COORD_BIN
    EOS_BIN = COORD_BIN + 1
    PAD_BIN = COORD_BIN + 2
    MAX_VERT_LENGTH = 800 + 2
    START_FACE_VERT_IDX = 0
    NEXT_FACE_VERT_IDX = 1
    STOP_FACE_VERT_IDX = 2
    FACE_VERT_OFFSET = STOP_FACE_VERT_IDX + 1
    MAX_FACE_INDICES = 2800

    def __init__(self, dataset_list_path, mode):
        dataset_list_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
        dataset_list_path = os.path.join(dataset_list_dir, dataset_list_path)
        self.mode = mode
        with open(dataset_list_path, 'r') as f:
            self.dataset_list = f.readlines()
        self.pad_id = self.PAD_BIN
        self.graphpool = FaceGraphPool(n=self.MAX_VERT_LENGTH, m=self.MAX_FACES_INDICES)

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        obj_file = self.dataset_list[idx].strip()
        print (idx)
        try:
            verts, faces = preprocess_mesh_obj(obj_file, self.COORD_BIN)
        except:
            # If fail, try another random example
            to_try = np.random.randint(0, len(self.dataset_list))
            return self.__getitem__(to_try)

        # Flattern verts, order Y(up), X(front), Z(right)
        reordered_verts = np.zeros_like(verts)
        reordered_verts[:,0] = verts[:,1]
        reordered_verts[:,1] = verts[:,0]
        reordered_verts[:,2] = verts[:,2]
        # pad start and end bin at the front
        st_end_verts = np.array([[self.INIT_BIN] * 3, [self.EOS_BIN] * 3])
        full_verts = np.concatenate([st_end_verts, reordered_verts], axis=0).astype(np.int64)

        if full_verts.shape[0] > self.MAX_VERT_LENGTH:
            to_try = np.random.randint(0, len(self.dataset_list))
            return self.__getitem__(to_try)
        
        faces += self.FACE_VERT_OFFSET
        flattern_faces = [self.START_FACE_VERT_IDX] + faces.flatten().astype(np.int64).tolist() + [self.STOP_FACE_VERT_IDX]
        # -1 for considering the FACE_EOS_BIN
        if len(flattern_faces) > self.MAX_FACE_INDICES:
            to_try = np.random.randint(0, len(self.dataset_list))
            return self.__getitem__(to_try)
        
        return FaceDataTuple(g=self.graphpool.get_graph_for_size(len(full_verts), len(faces)+1), 
                 num_edges_enc=self.graphpool.num_edges['enc'](len(full_verts), len(faces)+1),
                 num_edges_ed=self.graphpool.num_edges['ed'](len(full_verts), len(faces)+1),
                 num_edges_dd=self.graphpool.num_edges['dd'](len(full_verts), len(faces)+1),
                 src=full_verts, tgt=flattern_faces, mode=self.mode, device='cpu')


# different collate function for infer and train?
def collate_facegraphs(data):
    """Batching a list of datapoints for dataloader.
    Parameters
    ----------
    data : list of 8-tuples.
       Each tuples contains a graph, num_edges_enc, num_edges_ed, num_edges_dd,
       src sequence, tgt sequence, mode, device 
    Returns
    -------
    g : tuple of batched DGLGraph and input tensors
    """
    print ('hahaha')
    assert len(data[0]) == 8, \
            'Expect the tuple to be of length 5, got {:d}'.format(len(data[0]))
    g_list, num_edges_enc, num_edges_ed, num_edges_dd, src_buf, tgt_buf, modes, devices = map(list, zip(*data))
    num_edges = {'enc': num_edges_enc, 'ed': num_edges_ed, 'dd': num_edges_dd}
    src_lens = [len(_) for _ in src_buf]
    tgt_lens = [len(_) - 1 for _ in tgt_buf]
    mode = modes[0]
    device = devices[0]
    g = dgl.batch(g_list)
    src, tgt, tgt_y = [], [], []
    src_pos, tgt_pos = [], []
    enc_ids, dec_ids = [], []
    e2e_eids, d2d_eids, e2d_eids = [], [], []
    n_nodes, n_edges, n_tokens = 0, 0, 0
    n_enc_nodes = 0

    for src_sample, tgt_sample, n, m, n_ee, n_ed, n_dd in zip(src_buf, tgt_buf, src_lens, tgt_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
        src.append(th.tensor(src_sample, dtype=th.long, device=device))
        # Add the tgt with current n_node for forward indexing
        tgt_sample = np.array(tgt_sample)
        indexed_tgt_sample = tgt_sample + n_enc_nodes
        n_enc_nodes += n
        tgt.append(th.tensor(indexed_tgt_sample[:-1], dtype=th.long, device=device))
        tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
        src_pos.append(th.arange(n, dtype=th.long, device=device))
        tgt_pos.append(th.arange(m, dtype=th.long, device=device))
        enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
        n_nodes += n
        dec_ids.append(th.arange(n_nodes, n_nodes + m, dtype=th.long, device=device))
        n_nodes += m
        e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
        n_edges += n_ee
        e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
        n_edges += n_ed
        d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
        n_edges += n_dd
        n_tokens += m

    g.set_n_initializer(dgl.init.zero_initializer)
    g.set_e_initializer(dgl.init.zero_initializer)

    return FaceGraph(g=g,
                 src=(th.cat(src), th.cat(src_pos)),
                 tgt=(th.cat(tgt), th.cat(tgt_pos)),
                 tgt_y=th.cat(tgt_y),
                 nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                 eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                 nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                 n_nodes=n_nodes,
                 n_edges=n_edges,
                 n_tokens=n_tokens)
