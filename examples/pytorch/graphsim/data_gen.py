import taichi_88 as sim2d
import argparse
import numpy as np
import os

argparser = argparse.ArgumentParser()

argparser.add_argument('--num_traj',type=int,default=20)
argparser.add_argument('--steps',type=int,default=200)
argparser.add_argument('--hist_len',type=int,default=5)

args = argparser.parse_args()

pos = sim2d.reset()
gui = sim2d.init_render()

class Hist:
    def __init__(self,hist_len):
        self.hist_len = hist_len
        self.reset()

    def reset(self):
        self.pos_hist = [None for i in range(self.hist_len)]
        self.vel_hist = [None for i in range(self.hist_len)]
        self.init = True

    def __call__(self,pos):
        if self.init:
            for i in range(self.hist_len):
                self.pos_hist[i] = pos.copy()
                self.vel_hist[i] = np.zeros_like(pos).astype('float')
            self.init = False

        else:
            self.pos_hist.pop()
            self.pos_hist.insert(0,pos)
            self.vel_hist.pop()
            self.vel_hist.insert(0,pos-self.pos_hist[1])

        return np.hstack(self.vel_hist)

if not os.path.exists('data'):
    os.mkdir('data')

hist_buffer = Hist(args.hist_len)
for stage in [('train',args.num_traj),('valid',int(args.num_traj/2)),('test',int(args.num_traj/4))]:
    src_list = []
    label_list = []
    hist_list = []
    for traj in range(stage[1]):
        traj_list = []
        traj_hist_list = [] 
        hist_buffer.reset()
        for step in range(args.steps):
            pos = sim2d.step()
            traj_list.append(pos)
            traj_hist_list.append(hist_buffer(pos))
        print(stage[0],'Traj: ',traj)
        hist_list= hist_list+ traj_hist_list[:-1] # Align 
        src_list = src_list + traj_list[:-1]
        label_list = label_list + traj_list[1:]
    np.savez('data/mpm2d_water_{}.npz'.format(stage[0]),
             node_state=src_list,
             node_velocity=hist_list,
             label_state=label_list)
    print(len(label_list))

print("File has been saved!")