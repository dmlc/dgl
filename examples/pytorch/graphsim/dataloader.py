import torch
import dgl
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import copy
from sklearn.neighbors import radius_neighbors_graph
import networkx as nx
# Need to define the pytorch data loader for point cloud data
# Since this can enable GPU training, significant acceleration is expected

# The graph on the graph net should be bidirectional
def build_dense_graph(n_particles):
    g = nx.complete_graph(n_particles)
    return dgl.from_networkx(g)

class TaichiDataset(Dataset):
    def __init__(self,path):
        self.path = path
        zipfile = np.load(self.path)
        self.node_state    = zipfile['node_state']
        self.node_velocity = zipfile['node_velocity']
        self.label_state   = zipfile['label_state']
        self.vel_mean = torch.from_numpy(zipfile['vel_mean'])
        self.vel_std  = torch.from_numpy(zipfile['vel_std'])
        self.target_acc = zipfile['target_acc']
        self.acc_mean = torch.from_numpy(zipfile['acc_mean'])
        self.acc_std  = torch.from_numpy(zipfile['acc_std'])

    def __len__(self):
        return self.node_state.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        src_frame = self.node_state[idx,:,:]
        src_vels  = self.node_velocity[idx,:,:]
        tar_frame = self.label_state[idx,:,:]
        target_acc= self.target_acc[idx,:,:]
        return (src_frame,src_vels,tar_frame,target_acc)

class TaichiTrainDataset(TaichiDataset):
    def __init__(self):
        super(TaichiTrainDataset,self).__init__('data/mpm2d_water_train.npz')
        self.n_particles = 2048
        self.dim = 2
        self.dt = 2e-4
        self.substeps = 50
        self.boundary = np.array([0.966,0.0325,0.966,0.0325])

class TaichiValidDataset(TaichiDataset):
    def __init__(self):
        super(TaichiValidDataset,self).__init__('data/mpm2d_water_valid.npz')

class TaichiTestDataset(TaichiDataset):
    def __init__(self):
        super(TaichiTestDataset,self).__init__('data/mpm2d_water_test.npz')

class MultiBodyDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.zipfile = np.load(self.path)
        self.node_state = self.zipfile['data']
        self.node_label = self.zipfile['label']
        self.n_particles= self.zipfile['n_particles']

    def __len__(self):
        return self.node_state.shape[0]

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        node_state = self.node_state[idx,:,:]
        node_label = self.node_label[idx,:,:]
        return (node_state,node_label)

class MultiBodyTrainDataset(MultiBodyDataset):
    def __init__(self):
        super(MultiBodyTrainDataset,self).__init__('./data/n_body_train.npz')
        self.stat_median = self.zipfile['median']
        self.stat_max    = self.zipfile['max']
        self.stat_min    = self.zipfile['min']

class MultiBodyValidDataset(MultiBodyDataset):
    def __init__(self):
        super(MultiBodyValidDataset,self).__init__('./data/n_body_valid.npz')

class MultiBodyTestHalfDataset(MultiBodyDataset):
    def __init__(self):
        super(MultiBodyTestHalfDataset,self).__init__('./data/n_body_halftest.npz')

class MultiBodyTestFullDataset(MultiBodyDataset):
    def __init__(self):
        super(MultiBodyTestFullDataset,self).__init__('./data/n_body_fulltest.npz')

class MultiBodyTestDoubDataset(MultiBodyDataset):
    def __init__(self):
        super(MultiBodyTestDoubDataset,self).__init__('./data/n_body_doubtest.npz')

# Input idx and numpy array, output 
class TaichiGraphCollator:
    def __init__(self,radius,self_loop=True):
        self.radius = radius
        self.self_loop = self_loop

    def __call__(self,batch): # [(src_frame,src_vels,tar_frame),...]
        src_graph_list = []
        src_vels  = [] 
        src_coord = []
        dst_coord = []
        dst_acc   = []
        for frame in batch:
            src_graph = radius_neighbors_graph(frame[0],
                                           radius=self.radius,
                                           include_self=self.self_loop)

            src_graph = dgl.from_scipy(src_graph)
            src_graph_list.append(src_graph)
            src_coord.append(torch.from_numpy(frame[0]))
            src_vels.append(torch.from_numpy(frame[1]))
            dst_coord.append(torch.from_numpy(frame[2]))
            dst_acc.append(torch.from_numpy(frame[3]))

        src_batch_g = dgl.batch(src_graph_list)
        src_coord = torch.vstack(src_coord)
        src_vels  = torch.vstack(src_vels)
        tar_coord = torch.vstack(dst_coord)
        dst_acc   = torch.vstack(dst_acc)
        return src_batch_g, src_coord, src_vels, tar_coord, dst_acc

# Construct fully connected graph
class MultiBodyGraphCollator:
    def __init__(self,n_particles):
        self.n_particles = n_particles
        self.graph = dgl.from_networkx(nx.complete_graph(self.n_particles))

    def __call__(self,batch):
        graph_list = [] # Actually we don't need to build a graph each time, for API uniform
        data_list = []
        label_list = []
        for frame in batch:
            #graph_list.append(build_dense_graph(n_particles))
            graph_list.append(copy.deepcopy(self.graph))
            data_list.append(torch.from_numpy(frame[0]))
            label_list.append(torch.from_numpy(frame[1]))

        graph_batch = dgl.batch(graph_list)
        data_batch  = torch.vstack(data_list)
        label_batch = torch.vstack(label_list)
        return graph_batch, data_batch, label_batch


if __name__ == '__main__':
    ds = MultiBodyTrainDataset()
    collator = MultiBodyGraphCollator(ds.n_particles)
    dataloader = DataLoader(ds,batch_size=4,shuffle=True,num_workers=0,collate_fn=collator)
    for graph_b,data_b,label_b in dataloader:
        print(graph_b)
        print(data_b.shape)
        print(label_b.shape)
        break
        