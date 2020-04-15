import dgl
import networkx as nx
import random
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from load_data import *
from utils import *
from model import *
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description='STGCN_WAVE')
parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--seed', default=2333, type=int, help='seed')
parser.add_argument('--disablecuda', action='store_true', help='Disable CUDA')
parser.add_argument('--batch_size', type=int, default=50, help='batch size for training and validation (default: 50)')
parser.add_argument('--epochs', type=int, default=50, help='epochsfor training  (default: 50)')
args = parser.parse_args()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda") if torch.cuda.is_available() and not args.disablecuda else torch.device("cpu")

def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    """
    :param distance_df: data frame with three columns: [from, to, distance].
    :param sensor_ids: list of sensor ids.
    :param normalized_k: entries that become lower than normalized_k after normalization are set to zero for sparsity.
    :return:
    """
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    # Builds sensor id to index map.
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i

    # Fills cells in the matrix with distances.
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # Calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()
    std = distances.std()
    adj_mx = np.exp(-np.square(dist_mx / std))
    # Make the adjacent matrix symmetric by taking the max.
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # Sets entries that lower than a threshold, i.e., k, to zero for sparsity.
    adj_mx[adj_mx < normalized_k] = 0
    return sensor_ids, sensor_id_to_ind, adj_mx

with open('./data/sensor_graph/graph_sensor_ids.txt') as f:
    sensor_ids = f.read().strip().split(',')
distance_df = pd.read_csv('./data/sensor_graph/distances_la_2012.csv', dtype={'from': 'str', 'to': 'str'})

_, sensor_id_to_ind, adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
ng = nx.from_numpy_matrix(adj_mx)
G = dgl.DGLGraph(ng)

df = pd.read_hdf('./data/metr-la.h5')
num_samples, num_nodes = df.shape

a = df.to_numpy()


n_his = 144

save_path = "save/stgcnwavemodel.pt"


day_slot = 288
n_train, n_val, n_test = 34, 5, 5


n_pred = 3
n_route = 207
blocks = [1, 16, 32, 64, 32, 128]
drop_prob = 0


batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


W = adj_mx
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[: len_train]
val = df[len_train: len_train + len_val]
test = df[len_train + len_val:]

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


x_train, y_train = data_transform(train, n_his, n_pred, day_slot, device)
x_val, y_val = data_transform(val, n_his, n_pred, day_slot, device)
x_test, y_test = data_transform(test, n_his, n_pred, day_slot, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


loss = nn.MSELoss()
model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob).to(device)
optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

min_val_loss = np.inf
for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0
    model.train()
    for x, y in train_iter:
        y_pred = model(x).view(len(x), -1)
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)
    if val_loss < min_val_loss:
        min_val_loss = val_loss
        torch.save(model.state_dict(), save_path)
    print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)

    
best_model = STGCN_WAVE(blocks, n_his, n_route, G, drop_prob).to(device)
best_model.load_state_dict(torch.load(save_path))


l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
