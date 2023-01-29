import argparse
import random

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
from load_data import *
from model import *
from sensors2graph import *
from sklearn.preprocessing import StandardScaler
from utils import *

import dgl

parser = argparse.ArgumentParser(description="STGCN_WAVE")
parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
parser.add_argument("--disablecuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "--batch_size",
    type=int,
    default=50,
    help="batch size for training and validation (default: 50)",
)
parser.add_argument(
    "--epochs", type=int, default=50, help="epochs for training  (default: 50)"
)
parser.add_argument(
    "--num_layers", type=int, default=9, help="number of layers"
)
parser.add_argument("--window", type=int, default=144, help="window length")
parser.add_argument(
    "--sensorsfilepath",
    type=str,
    default="./data/sensor_graph/graph_sensor_ids.txt",
    help="sensors file path",
)
parser.add_argument(
    "--disfilepath",
    type=str,
    default="./data/sensor_graph/distances_la_2012.csv",
    help="distance file path",
)
parser.add_argument(
    "--tsfilepath", type=str, default="./data/metr-la.h5", help="ts file path"
)
parser.add_argument(
    "--savemodelpath",
    type=str,
    default="stgcnwavemodel.pt",
    help="save model path",
)
parser.add_argument(
    "--pred_len",
    type=int,
    default=5,
    help="how many steps away we want to predict",
)
parser.add_argument(
    "--control_str",
    type=str,
    default="TNTSTNTST",
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
parser.add_argument(
    "--channels",
    type=int,
    nargs="+",
    default=[1, 16, 32, 64, 32, 128],
    help="model strcture controller, T: Temporal Layer, S: Spatio Layer, N: Norm Layer",
)
args = parser.parse_args()

device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not args.disablecuda
    else torch.device("cpu")
)

with open(args.sensorsfilepath) as f:
    sensor_ids = f.read().strip().split(",")

distance_df = pd.read_csv(args.disfilepath, dtype={"from": "str", "to": "str"})

adj_mx = get_adjacency_matrix(distance_df, sensor_ids)
sp_mx = sp.coo_matrix(adj_mx)
G = dgl.from_scipy(sp_mx)


df = pd.read_hdf(args.tsfilepath)
num_samples, num_nodes = df.shape

tsdata = df.to_numpy()


n_his = args.window

save_path = args.savemodelpath


n_pred = args.pred_len
n_route = num_nodes
blocks = args.channels
# blocks = [1, 16, 32, 64, 32, 128]
drop_prob = 0
num_layers = args.num_layers

batch_size = args.batch_size
epochs = args.epochs
lr = args.lr


W = adj_mx
len_val = round(num_samples * 0.1)
len_train = round(num_samples * 0.7)
train = df[:len_train]
val = df[len_train : len_train + len_val]
test = df[len_train + len_val :]

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)


x_train, y_train = data_transform(train, n_his, n_pred, device)
x_val, y_val = data_transform(val, n_his, n_pred, device)
x_test, y_test = data_transform(test, n_his, n_pred, device)

train_data = torch.utils.data.TensorDataset(x_train, y_train)
train_iter = torch.utils.data.DataLoader(train_data, batch_size, shuffle=True)
val_data = torch.utils.data.TensorDataset(x_val, y_val)
val_iter = torch.utils.data.DataLoader(val_data, batch_size)
test_data = torch.utils.data.TensorDataset(x_test, y_test)
test_iter = torch.utils.data.DataLoader(test_data, batch_size)


loss = nn.MSELoss()
G = G.to(device)
model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
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
    print(
        "epoch",
        epoch,
        ", train loss:",
        l_sum / n,
        ", validation loss:",
        val_loss,
    )


best_model = STGCN_WAVE(
    blocks, n_his, n_route, G, drop_prob, num_layers, device, args.control_str
).to(device)
best_model.load_state_dict(torch.load(save_path))


l = evaluate_model(best_model, loss, test_iter)
MAE, MAPE, RMSE = evaluate_metric(best_model, test_iter, scaler)
print("test loss:", l, "\nMAE:", MAE, ", MAPE:", MAPE, ", RMSE:", RMSE)
