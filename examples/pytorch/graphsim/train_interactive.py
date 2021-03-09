import taichi_88 as sim2d
from models import OnlinePrepareLayer,InteractionGNN
import torch
import torch.nn as nn
from sklearn.neighbors import radius_neighbors_graph
import dgl
import time
import copy
import numpy as np

prep = OnlinePrepareLayer(sim2d.n_particles,
                   sim2d.dim,
                   5,
                   torch.from_numpy(sim2d.boundary),
                   {'vel_mean':0,'vel_std':1},
                   0.04)

ignn = InteractionGNN(10,
                      14,
                      3,
                      20,
                      10,
                      2,
                      2)

pos = sim2d.reset()
gui = sim2d.init_render()

optim = torch.optim.Adam(ignn.parameters(),lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(20):
    pos = sim2d.reset()
    prev_pos = pos.copy()
    prep.reset()
    pc_list = []
    for i in range(200):
        optim.zero_grad()
        g_sci = radius_neighbors_graph(pos,0.03,include_self=True)
        
        g = dgl.from_scipy(g_sci)
        pc_list.append(pos)
        in_pos = torch.from_numpy(pos)
        t = time.time()
        node_feats,edge_feats = prep(g,in_pos)
        pred_a = ignn(g,node_feats,edge_feats)
        # Update velocity
        # Consider gravity as prior
        pred_a[:,1] = pred_a[:,1] - 9.8
        pred_v = torch.from_numpy(pos-prev_pos) + pred_a*sim2d.dt*sim2d.substeps
        pred_pos = in_pos + pred_v*sim2d.dt*sim2d.substeps
        prev_pos = pos
        pos = sim2d.step()
        loss = loss_fn(pred_pos,torch.from_numpy(pos))
        loss.backward()
        optim.step()
        sim2d.render(pos,gui)
    np.savez('test_size.npz',pc_list)
    in_pos = torch.from_numpy(sim2d.reset())
    prev_pos = copy.deepcopy(in_pos)
    prep.reset()
    for i in range(50):
        with torch.no_grad():
            sim2d.render(in_pos.numpy(),gui)
            g_sci = radius_neighbors_graph(in_pos.numpy(),0.03,include_self=True)
            g = dgl.from_scipy(g_sci)
            node_feats,edge_feats = prep(g,in_pos)
            pred_a = ignn(g,node_feats,edge_feats)
            pred_a[:,1] = pred_a[:,1] - 9.8
            print(pred_a.max())
            pred_v = (in_pos-prev_pos) + pred_a*sim2d.dt*sim2d.substeps
            prev_pos = in_pos
            in_pos = in_pos + pred_v*sim2d.dt*sim2d.substeps
            

