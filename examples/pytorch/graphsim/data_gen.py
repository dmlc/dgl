import taichi_88 as sim2d
import argparse
import numpy as np
import os

argparser = argparse.ArgumentParser()

argparser.add_argument('--num_traj',type=int,default=20)
argparser.add_argument('--steps',type=int,default=200)
argparser.add_argument('--hist_len',type=int,default=5)
argparser.add_argument('--data_path',type=str,default='data')
argparser.add_argument('--order',type=str,default='first',
                       help='Order of Dynamic Approximated first means we have real velocity,\
                       second means we have real position only')

args = argparser.parse_args()

pos = sim2d.reset()

class Hist:
    def __init__(self,hist_len,order):
        self.hist_len = hist_len
        self.order = order
        self.vel_mean = []
        self.vel_std  = []
        # Use to compute mean and variance of velocity in the particle set
        self.vel_running_mean = 0
        self.vel_running_sqmean = 0
        self.reset()

    def reset(self):
        self.pos_hist = [None for i in range(self.hist_len)]
        self.vel_hist = [None for i in range(self.hist_len)]
        self.acc = None
        self.vel_running_mean  = 0
        self.vel_running_sqmean= 0
        self.init = True

    def __call__(self, pos, vel=None):
        if self.init:
            for i in range(self.hist_len):
                self.pos_hist[i] = pos.copy()
                self.vel_hist[i] = np.zeros_like(pos).astype('float')
            self.init = False
            acc = self.vel_hist[0] - self.vel_hist[1]
        else:
            self.pos_hist.pop()
            self.pos_hist.insert(0,pos)
            self.vel_hist.pop()
            if self.order == 'second':
                self.vel_hist.insert(0,pos-self.pos_hist[1])
            else:
                self.vel_hist.insert(0,vel)
            
            acc = self.vel_hist[0] - self.vel_hist[1]
            #print(acc.shape)

        return np.hstack(self.vel_hist),acc

if not os.path.exists('data'):
    os.mkdir('data')

hist_buffer = Hist(args.hist_len,args.order)
for stage in [('train',args.num_traj),('valid',int(args.num_traj/2)),('test',int(args.num_traj/4))]:
    src_list = []
    label_list = []
    hist_list = []
    target_acc =[]
    for traj in range(stage[1]):
        traj_list = []
        traj_hist_list = []
        traj_acc = []
        hist_buffer.reset()
        for step in range(args.steps):
            pos = sim2d.step()
            vel = sim2d.get_v()
            hist_buffer.vel_running_mean  = hist_buffer.vel_running_mean*(step/(step+1))+1/(step+1)*vel.mean(axis=0)
            hist_buffer.vel_running_sqmean= hist_buffer.vel_running_sqmean*(step/(step+1))+1/(step+1)*(vel**2).mean(axis=0)
            traj_list.append(pos)
            vel_hist,acc = hist_buffer(pos,vel if args.order=='first' else None)
            traj_hist_list.append(vel_hist)
            traj_acc.append(acc)
            #print(acc.shape)
        print(stage[0],'Traj: ',traj)
        target_acc = target_acc + traj_acc[:-1]   # Align
        hist_list= hist_list+ traj_hist_list[:-1] # Align
        src_list = src_list + traj_list[:-1]
        label_list = label_list + traj_list[1:]
        hist_buffer.vel_mean.append(hist_buffer.vel_running_mean)
        hist_buffer.vel_std.append(np.sqrt(hist_buffer.vel_running_sqmean-hist_buffer.vel_running_mean**2))
    vel_mean = np.array(hist_buffer.vel_mean).mean(axis=0)
    vel_std  = np.array(hist_buffer.vel_std).mean(axis=0)

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    acc = np.vstack(target_acc)
    acc_mean = acc.mean()
    acc_std  = acc.std()
    np.savez('{0}/mpm2d_water_{1}.npz'.format(args.data_path,stage[0]),
             node_state=src_list,
             node_velocity=hist_list,
             label_state=label_list,
             vel_mean=vel_mean,
             vel_std =vel_std,
             target_acc=target_acc,
             acc_mean=acc_mean,
             acc_std =acc_std)

print("File has been saved!")