from __future__ import print_function, division
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from .vertexgraph import *
from .facegraph import *

VertexDataTuple = namedtuple('VertexDataTuple', ['g', 'num_edges_dd', 'tgt', 'mode', 'device'])
VertexGraph = namedtuple('Graph',
                   ['g', 'tgt', 'tgt_y', 'nids', 'eids',  'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])


class VertexDataset(Dataset):
    """Vertex Dataset"""
    COORD_BIN = 32
    INIT_BIN = COORD_BIN
    EOS_BIN = COORD_BIN + 1
    PAD_BIN = COORD_BIN + 2
    MAX_VERT_LENGTH = 98
    MAX_LENGTH = MAX_VERT_LENGTH * 3 + 1
    
    def __init__(self, file_list_path, mode):
        dataset_list_dir = '/home/ubuntu/data/new/ShapeNetCore.v2/'
        dataset_list_path = os.path.join(dataset_list_dir, dataset_list_file)
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
        obj_file = dataset_list[idx].strip()
        try:
            verts, faces = preprocess_mesh_obj(obj_file)
        except:
           # If fail, try another random example
           return self.__getitem__(self, np.random.randint(0, len(self.dataset_list))
            
        # Flattern verts, order Y(up), X(front), Z(right)
        reordered_verts = np.zeros_like(verts)
        reordered_verts[:,0] = verts[:,1]
        reordered_verts[:,1] = verts[:,0]
        reordered_verts[:,2] = verts[:,2]
        flattern_verts = [self.INIT_BIN] + reordered_verts.flatten().astype(np.int64).tolist() + [self.EOS_BIN]
        # exp
        if len(flattern_verts) > self.MAX_LENGTH:
           return self.__getitem__(self, np.random.randint(0, len(self.dataset_list))
        flattern_verts = flattern_verts[:self.MAX_LENGTH-1] + [self.EOS_BIN]
        return VertexData(g=self.graph_pool.g_pool[len(flattern_verts)-1], 
                num_edges_dd=self.graph_pool.num_edges['dd'][len(flattern_verts)-1],
                tgt=falltern_verts, mode=self.mode, device='cpu')

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
    assert len(data[0]) == 5, \
            'Expect the tuple to be of length 5, got {:d}'.format(len(data[0]))
    g_list, num_edges_dd, tgt_buf, modes, devices = map(list, zip(*data))
    num_edges = {'dd': num_edges_dd}
    mode = modes[0]
    device = device[0]
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
        d2d_eids.append(th.arange(n_edges, n_edges + n_dd['dd'], dtype=th.long, device=device))
        n_nodes += n
        n_tokens += n
        n_edges += n_dd['dd']

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


if __name__ == '__main__':
    np.random.seed(1111)
    train_dataset = VertexDataset()
    def __init__(self, file_list_path, 'train'):
    trainset = DataLoader(dataset=)
