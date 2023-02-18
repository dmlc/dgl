import argparse
import os
import pickle
import time

import dgl

import numpy as np
import torch
import torch.optim as optim
from dataset import LanderDataset
from models import LANDER

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--levels", type=str, default="1")
parser.add_argument("--faiss_gpu", action="store_true")
parser.add_argument("--model_filename", type=str, default="lander.pth")

# KNN
parser.add_argument("--knn_k", type=str, default="10")

# Model
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--num_conv", type=int, default=4)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--gat", action="store_true")
parser.add_argument("--gat_k", type=int, default=1)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--use_cluster_feat", action="store_true")
parser.add_argument("--use_focal_loss", action="store_true")

# Training
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-5)

args = parser.parse_args()

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


##################
# Data Preparation
def prepare_dataset_graphs(data_path, k_list, lvl_list):
    with open(data_path, "rb") as f:
        features, labels = pickle.load(f)
    gs = []
    for k, l in zip(k_list, lvl_list):
        dataset = LanderDataset(
            features=features,
            labels=labels,
            k=k,
            levels=l,
            faiss_gpu=args.faiss_gpu,
        )
        gs += [g.to(device) for g in dataset.gs]
    return gs


k_list = [int(k) for k in args.knn_k.split(",")]
lvl_list = [int(l) for l in args.levels.split(",")]
gs = prepare_dataset_graphs(args.data_path, k_list, lvl_list)
test_gs = prepare_dataset_graphs(args.test_data_path, k_list, lvl_list)

##################
# Model Definition
feature_dim = gs[0].ndata["features"].shape[1]
model = LANDER(
    feature_dim=feature_dim,
    nhid=args.hidden,
    num_conv=args.num_conv,
    dropout=args.dropout,
    use_GAT=args.gat,
    K=args.gat_k,
    balance=args.balance,
    use_cluster_feat=args.use_cluster_feat,
    use_focal_loss=args.use_focal_loss,
)
model = model.to(device)
model.train()
best_model = None
best_loss = np.Inf

#################
# Hyperparameters
opt = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.epochs, eta_min=1e-5
)

###############
# Training Loop
for epoch in range(args.epochs):
    all_loss_den_val = 0
    all_loss_conn_val = 0
    for g in gs:
        opt.zero_grad()
        g = model(g)
        loss, loss_den_val, loss_conn_val = model.compute_loss(g)
        all_loss_den_val += loss_den_val
        all_loss_conn_val += loss_conn_val
        loss.backward()
        opt.step()
    scheduler.step()
    print(
        "Training, epoch: %d, loss_den: %.6f, loss_conn: %.6f"
        % (epoch, all_loss_den_val, all_loss_conn_val)
    )
    # Report test
    all_test_loss_den_val = 0
    all_test_loss_conn_val = 0
    with torch.no_grad():
        for g in test_gs:
            g = model(g)
            loss, loss_den_val, loss_conn_val = model.compute_loss(g)
            all_test_loss_den_val += loss_den_val
            all_test_loss_conn_val += loss_conn_val
    print(
        "Testing, epoch: %d, loss_den: %.6f, loss_conn: %.6f"
        % (epoch, all_test_loss_den_val, all_test_loss_conn_val)
    )
    if all_test_loss_conn_val + all_test_loss_den_val < best_loss:
        best_loss = all_test_loss_conn_val + all_test_loss_den_val
        print("New best epoch", epoch)
        torch.save(model.state_dict(), args.model_filename + "_best")
    torch.save(model.state_dict(), args.model_filename)

torch.save(model.state_dict(), args.model_filename)
