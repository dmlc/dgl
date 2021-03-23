from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np
from math import sin, cos, radians, pi
import argparse

'''
This file comes from https://github.com/jsikyoon/Interaction-networks_tensorflow
Which generate Multi-body Dynamic simulation data used for Interaction network
'''

# 5 features on the state [mass,x,y,x_vel,y_vel]
fea_num = 5
# G
#G = 6.67428e-11;
G = 10**5
# time step
diff_t = 0.001


def init(total_state, n_body, fea_num, orbit):
    data = np.zeros((total_state, n_body, fea_num), dtype=float)
    if(orbit):
        data[0][0][0] = 100
        data[0][0][1:5] = 0.0
        # The position are initialized randomly.
        for i in range(1, n_body):
            data[0][i][0] = np.random.rand()*8.98+0.02
            distance = np.random.rand()*90.0+10.0
            theta = np.random.rand()*360
            theta_rad = pi/2 - radians(theta)
            data[0][i][1] = distance*cos(theta_rad)
            data[0][i][2] = distance*sin(theta_rad)
            data[0][i][3] = -1*data[0][i][2]/norm(data[0][i][1:3])*(
                G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000
            data[0][i][4] = data[0][i][1]/norm(data[0][i][1:3])*(
                G*data[0][0][0]/norm(data[0][i][1:3])**2)*distance/1000
            # data[0][i][3]=np.random.rand()*10.0-5.0;
            # data[0][i][4]=np.random.rand()*10.0-5.0;
    else:
        for i in range(n_body):
            data[0][i][0] = np.random.rand()*8.98+0.02
            distance = np.random.rand()*90.0+10.0
            theta = np.random.rand()*360
            theta_rad = pi/2 - radians(theta)
            data[0][i][1] = distance*cos(theta_rad)
            data[0][i][2] = distance*sin(theta_rad)
            data[0][i][3] = np.random.rand()*6.0-3.0
            data[0][i][4] = np.random.rand()*6.0-3.0
    return data


def norm(x):
    return np.sqrt(np.sum(x**2))


def get_f(reciever, sender):
    diff = sender[1:3]-reciever[1:3]
    distance = norm(diff)
    if(distance < 1):
        distance = 1
    return G*reciever[0]*sender[0]/(distance**3)*diff

# Compute stat according to the paper for normalization
def compute_stats(train_curr):
    data = np.vstack(train_curr).reshape(-1,fea_num)
    stat_median = np.median(data,axis=0)
    stat_max    = np.max(data,axis=0)
    stat_min    = np.min(data,axis=0)
    return stat_median,stat_max,stat_min


def calc(cur_state, n_body):
    next_state = np.zeros((n_body, fea_num), dtype=float)
    f_mat = np.zeros((n_body, n_body, 2), dtype=float)
    f_sum = np.zeros((n_body, 2), dtype=float)
    acc = np.zeros((n_body, 2), dtype=float)
    for i in range(n_body):
        for j in range(i+1, n_body):
            if(j != i):
                f = get_f(cur_state[i][:3], cur_state[j][:3])
                f_mat[i, j] += f
                f_mat[j, i] -= f
        f_sum[i] = np.sum(f_mat[i], axis=0)
        acc[i] = f_sum[i]/cur_state[i][0]
        next_state[i][0] = cur_state[i][0]
        next_state[i][3:5] = cur_state[i][3:5]+acc[i]*diff_t
        next_state[i][1:3] = cur_state[i][1:3]+next_state[i][3:5]*diff_t
    return next_state

# The state is [mass,pos_x,pos_y,vel_x,vel_y]* n_body


def gen(n_body, num_steps,orbit):
    # initialization on just first state
    data = init(num_steps, n_body, fea_num, orbit)
    for i in range(1, num_steps):
        data[i] = calc(data[i-1], n_body)
    return data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--num_bodies',type=int,default=6)
    argparser.add_argument('--num_traj',type=int,default=10)
    argparser.add_argument('--steps',type=int,default=1000)
    argparser.add_argument('--data_path',type=str,default='data')

    args = argparser.parse_args()
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    # Generate training data
    # Need to generate src_target_pair for supervised learning
    train_curr = []
    train_next = []

    valid_curr = []
    valid_next = []

    test_halfbody_curr = []
    test_halfbody_next = []
    test_fullbody_curr = []
    test_fullbody_next = []
    test_doubbody_curr = []
    test_doubbody_next = []
    for i in range(args.num_traj):
        train_traj = gen(args.num_bodies,args.steps,True)

        train_curr.append(train_traj[:-1])
        train_next.append(train_traj[1:])
        print("Train Traj: ",i)
    
    # Label shall only contains
    stat_median,stat_max,stat_min = compute_stats(train_curr)
    np.savez(args.data_path+'/n_body_train.npz',
             data=np.vstack(train_curr),
             label=np.vstack(train_next)[:,:,[3,4]],
             n_particles = args.num_bodies,
             median=stat_median,
             max = stat_max,
             min = stat_min)

    for i in range(args.num_traj//2):
        valid_traj = gen(args.num_bodies,args.steps,True)
        valid_curr.append(valid_traj[:-1])
        valid_next.append(valid_traj[1:])
        print("Valid Traj: ",i)
    np.savez(args.data_path+"/n_body_valid.npz",
             data=np.vstack(valid_curr),
             label=np.vstack(valid_next)[:,:,[3,4]],
             n_particles = args.num_bodies)

    for i in range(args.num_traj//4):
        test_h_traj = gen(args.num_bodies//2,args.steps,True)
        test_f_traj = gen(args.num_bodies,args.steps,True)
        test_d_traj = gen(args.num_bodies*2,args.steps,True)
        
        test_halfbody_curr.append(test_h_traj[:-1])
        test_halfbody_next.append(test_h_traj[1:])
        test_fullbody_curr.append(test_f_traj[:-1])
        test_fullbody_next.append(test_f_traj[1:])
        test_doubbody_curr.append(test_d_traj[:-1])
        test_doubbody_next.append(test_d_traj[1:])
        
        print("Test Traj: ",i)

    np.savez(args.data_path+"/n_body_halftest.npz",
             data=np.vstack(test_halfbody_curr),
             label=np.vstack(test_halfbody_next)[:,:,[3,4]],
             n_particles = args.num_bodies//2)
    np.savez(args.data_path+"/n_body_fulltest.npz",
             data=np.vstack(test_fullbody_curr),
             label=np.vstack(test_fullbody_next)[:,:,[3,4]],
             n_particles = args.num_bodies)
    np.savez(args.data_path+"/n_body_doubtest.npz",
             data=np.vstack(test_doubbody_curr),
             label=np.vstack(test_doubbody_next)[:,:,[3,4]],
             n_particles = args.num_bodies*2)

    print("File has been saved!")