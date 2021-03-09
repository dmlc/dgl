import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.neighbors import radius_neighbors_graph

# Need to define the pytorch data loader for point cloud data
# Since this can enable GPU training, significant acceleration is expected

class TaichiDataset(Dataset):
    def __init__(self,path):
        self.path = path
        zipfile = np.load(self.path)
        self.node_state    = zipfile['node_state']
        self.node_velocity = zipfile['node_velocity']
        self.label_state   = zipfile['label_state']

    def __len__(self):
        return self.node_state.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        src_frame = self.node_state[idx,:,:]
        src_vels  = self.node_velocity[idx,:,:]
        tar_frame = self.label_state[idx,:,:]
        return (src_frame,src_vels,tar_frame)

class TaichiTrainDataset(TaichiDataset):
    def __init__(self):
        super(TaichiTrainDataset,self).__init__('data/mpm2d_water_train.npz')
        self.n_particles = 2048
        self.dim = 2
        self.dt = 2e-4
        self.substeps = 50
        self.boundary = np.array([0.966,0.0325,0.966,0.325])

class TaichiValidDataset(TaichiDataset):
    def __init__(self):
        super(TaichiValidDataset,self).__init__('data/mpm2d_water_valid.npz')

class TaichiTestDataset(TaichiDataset):
    def __init__(self):
        super(TaichiTestDataset,self).__init__('data/mpm2d_water_test.npz')

# Input idx and numpy array, output 
class GraphCollator:
    def __init__(self,radius,self_loop=True):
        self.radius = radius
        self.self_loop = self_loop

    def __call__(self,batch): # [(src_frame,src_vels,tar_frame),...]
        src_graph_list = []
        src_vels  = [] 
        src_coord = []
        dst_coord = []
        for frame in batch:
            src_graph = radius_neighbors_graph(frame[0],
                                           radius=self.radius,
                                           include_self=self.self_loop)

            src_graph = dgl.from_scipy(src_graph)
            src_graph_list.append(src_graph)
            src_coord.append(torch.from_numpy(frame[0]))
            dst_coord.append(torch.from_numpy(frame[2]))
            src_vels.append(torch.from_numpy(frame[1]))

        src_batch_g = dgl.batch(src_graph_list)
        src_coord = torch.vstack(src_coord)
        src_vels  = torch.vstack(src_vels)
        tar_coord = torch.vstack(dst_coord)
        return src_batch_g,src_coord,src_vels,tar_coord

if __name__ == '__main__':
    ds = TaichiTrainDataset()
    collator = GraphCollator(radius=0.03)
    dataloader = DataLoader(ds,batch_size=4,shuffle=True,num_workers=1,collate_fn=collator)
    for src_batch_g,src_coord,src_vels,tar_coord in dataloader:
        print(src_batch_g)
        print(src_coord.shape)
        print(tar_coord.shape)
        print(src_vels.shape)
        break
        