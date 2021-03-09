import taichi_88 as sim2d
from models import OnlinePrepareLayer,InteractionGNN
import torch
import torch.nn as nn
from sklearn.neighbors import radius_neighbors_graph
import dgl
import time
import copy
import numpy as np
import taichi as ti
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('--model_path',type=str,required=True)
argparser.add_argument('--video_path',type=str,default='video')
argparser.add_argument('--gpu',type=int,default=-1)
argparser.add_argument('--gui',action='store_true',default=False)
args = argparser.parse_args()

device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda:{}'.format(args.gpu))

prep = OnlinePrepareLayer(sim2d.n_particles,
                   sim2d.dim,
                   5,
                   torch.from_numpy(sim2d.boundary),
                   {'vel_mean':0,'vel_std':1},
                   0.04).to(device)

ignn = InteractionGNN(10,
                      14,
                      3,
                      20,
                      10,
                      2,
                      2)
ignn.load_state_dict(torch.load('saved_models/{}'.format(args.model_path)))
ignn = ignn.to(device)

pos = sim2d.reset()
if args.gui:
    gui = sim2d.init_render()

for epoch in range(50):
    pos = sim2d.reset()
    prep.reset()
    in_pos = torch.from_numpy(sim2d.reset())
    prev_pos = copy.deepcopy(in_pos)
    prep.reset()
    for i in range(50):
        with torch.no_grad():
            if args.gui:
                sim2d.render(in_pos.numpy(),gui)
            g_sci = radius_neighbors_graph(in_pos.numpy(),0.03,include_self=True)
            g = dgl.from_scipy(g_sci)
            node_feats,edge_feats = prep(g,in_pos)
            pred_a = ignn(g,node_feats,edge_feats)
            pred_a[:,1] = pred_a[:,1] - 9.8
            v = in_pos-prev_pos
            pred_v = v + pred_a*sim2d.dt*sim2d.substeps
            prev_pos = in_pos
            in_pos = in_pos + (pred_v+v)/2*sim2d.dt*sim2d.substeps
            
            

