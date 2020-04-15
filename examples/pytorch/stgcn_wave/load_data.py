import torch
import numpy as np
import pandas as pd


def load_matrix(file_path):
    return pd.read_csv(file_path, header=None).values.astype(float)


def load_data(file_path, len_train, len_val):
    df = pd.read_csv(file_path, header=None).values.astype(float)
    train = df[: len_train]
    val = df[len_train: len_train + len_val]
    test = df[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, day_slot, device):
    n_day = len(data) // day_slot
    n_route = data.shape[1]
    l = len(data)
    # print('n_day :',n_day)
    n_slot = day_slot - n_his - n_pred + 1
    num = l-n_his-n_pred
    # x = np.zeros([n_day * n_slot, 1, n_his, n_route])
    # y = np.zeros([n_day * n_slot, n_route])
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])
    
    cnt = 0
    for i in range(l-n_his-n_pred):
        head = i
        tail = i + n_his
        x[cnt, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
        y[cnt] = data[tail + n_pred - 1]
        cnt += 1
    # print('cnt :',cnt,'l :',l)
    # for i in range(n_day):
    #     for j in range(n_slot):
    #         t = i * n_slot + j
    #         s = i * day_slot + j
    #         e = s + n_his
    #         x[t, :, :, :] = data[s:e].reshape(1, n_his, n_route)
    #         y[t] = data[e + n_pred - 1]
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)
