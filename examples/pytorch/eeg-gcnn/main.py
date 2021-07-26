import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from sklearn.model_selection import train_test_split
from joblib import load
from EEGGraphConvNet import EEGGraphConvNet
from EEGGraphDataset import EEGGraphDataset
from dgl.dataloading import GraphDataLoader
from torch.utils.data import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import balanced_accuracy_score
from collections import Counter


if __name__ == "__main__":
    # argparse commandline args
    parser = argparse.ArgumentParser(description='Execute training pipeline on a given train/val subjects')
    parser.add_argument('--num_feats', type=int, default=6, help='Number of features per node for the graph')
    parser.add_argument('--num_nodes', type=int, default=8, help='Number of nodes in the graph')
    parser.add_argument('--gpu_idx', type=int, default=0,
                        help='index of GPU device that should be used for this run, defaults to 0.')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs used to train')
    parser.add_argument('--exp_name', type=str, default='default', help='Name for the test.')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch Size. Default is 512.')
    args = parser.parse_args()

    # use GPU when available
    _GPU_IDX = args.gpu_idx
    _DEVICE = torch.device(f'cuda:{_GPU_IDX}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(_DEVICE)
    print(f' Using device: {_DEVICE} {torch.cuda.get_device_name(_DEVICE)}')

    # retrieve inputs
    num_nodes = args.num_nodes
    _NUM_EPOCHS = args.num_epochs
    _EXPERIMENT_NAME = args.exp_name
    _BATCH_SIZE = args.batch_size
    num_feats = args.num_feats

    # set up input and targets from files
    memmap_x = f'norm_X'
    memmap_y = f'labels_y'
    x = load(memmap_x, mmap_mode='r')
    y = load(memmap_y, mmap_mode='r')

    # map 0/1 to diseased/healthy
    label_mapping, y = np.unique(y, return_inverse=True)
    print(f"Unique labels 0/1 mapping: {label_mapping}")

    # split the dataset to train and test. The ratio of test is 0.3.
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, stratify=y)

    # define model, optimizer, scheduler
    model = EEGGraphConvNet(num_feats)
    loss_function = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[i * 10 for i in range(1, 26)], gamma=0.1)

    num_trainable_params = np.sum([np.prod(p.size()) if p.requires_grad else 0 for p in model.parameters()])
    model = model.to(_DEVICE).double()

    # Dataloader========================================================================================================

    # use WeightedRandomSampler to balance the training dataset
    NUM_WORKERS = 8

    count = Counter(y_train)
    class_count = np.array([count[0], count[1]])

    weight = 1. / class_count

    samples_weights = np.array([weight[t] for t in y_train])
    samples_weights = torch.from_numpy(samples_weights)
    # provide weights for samples in the training set only

    # sampler needs to come up with training set size number of samples
    weighted_sampler = WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)

    # train data loader
    train_dataset = EEGGraphDataset(
        x=x_train, y=y_train, num_nodes=num_nodes
    )

    train_loader = GraphDataLoader(
        dataset=train_dataset, batch_size=_BATCH_SIZE,
        sampler=weighted_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # this loader is used without weighted sampling, to evaluate metrics on full training set after each epoch
    train_metrics_loader = GraphDataLoader(
        dataset=train_dataset, batch_size=_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # validation data loader
    valid_dataset = EEGGraphDataset(
        x=x_valid, y=y_valid, num_nodes=num_nodes
    )

    valid_loader = GraphDataLoader(
        dataset=valid_dataset, batch_size=_BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True
    )

    # initialize metric arrays
    auroc_train_history = []
    auroc_test_history = []
    balACC_train_history = []
    balACC_test_history = []
    loss_train_history = []
    loss_test_history = []

    # training=========================================================================================================
    for epoch in range(_NUM_EPOCHS):
        model.train()
        train_loss = []

        for batch_idx, batch in enumerate(train_loader):
            # send batch to GPU
            g, dataset_idx, y = batch
            g_batch = g.to(device=_DEVICE, non_blocking=True)
            y_batch = y.to(device=_DEVICE, non_blocking=True)
            optimizer.zero_grad()

            # forward pass
            outputs = model(g_batch)
            loss = loss_function(outputs, y_batch)
            train_loss.append(loss.item())

            # backward pass
            loss.backward()
            optimizer.step()

        # update learning rate
        scheduler.step()

    # evaluate model after each epoch for train-metric data============================================================
        model.eval()
        with torch.no_grad():
            y_probs_train = torch.empty(0, 2).to(_DEVICE)
            y_true_train, y_pred_train = [], []

            for i, batch in enumerate(train_metrics_loader):
                g, dataset_idx, y = batch
                g_batch = g.to(device=_DEVICE, non_blocking=True)
                y_batch = y.to(device=_DEVICE, non_blocking=True)

                # forward pass
                outputs = model(g_batch)

                _, predicted = torch.max(outputs.data, 1)
                y_pred_train += predicted.cpu().numpy().tolist()
                # concatenate along 0th dimension
                y_probs_train = torch.cat((y_probs_train, outputs.data), 0)
                y_true_train += y_batch.cpu().numpy().tolist()

        # returning prob distribution over target classes, take softmax over the 1st dimension
        y_probs_train = nn.functional.softmax(y_probs_train, dim=1).cpu().numpy()
        y_true_train = np.array(y_true_train)

    # evaluate model after each epoch for validation data ==============================================================
        y_probs_valid = torch.empty(0, 2).to(_DEVICE)
        y_true_valid, minibatch_loss, y_pred_valid = [], [], []

        for i, batch in enumerate(valid_loader):
            g, dataset_idx, y = batch
            g_batch = g.to(device=_DEVICE, non_blocking=True)
            y_batch = y.to(device=_DEVICE, non_blocking=True)

            # forward pass
            outputs = model(g_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_valid += predicted.cpu().numpy().tolist()

            loss = loss_function(outputs, y_batch)
            minibatch_loss.append(loss.item())
            y_probs_valid = torch.cat((y_probs_valid, outputs.data), 0)
            y_true_valid += y_batch.cpu().numpy().tolist()

        # returning prob distribution over target classes, take softmax over the 1st dimension
        y_probs_valid = torch.nn.functional.softmax(y_probs_valid, dim=1).cpu().numpy()
        y_true_valid = np.array(y_true_valid)

        # record training auroc and testing auroc
        auroc_train_history.append(roc_auc_score(y_true_train, y_probs_train[:, 1]))
        auroc_test_history.append(roc_auc_score(y_true_valid, y_probs_valid[:, 1]))

        # record training balanced accuracy and testing balanced accuracy
        balACC_train_history.append(balanced_accuracy_score(y_true_train, y_pred_train))
        balACC_test_history.append(balanced_accuracy_score(y_true_valid, y_pred_valid))

        # LOSS - epoch loss is defined as mean of minibatch losses within epoch
        loss_train_history.append(np.mean(train_loss))
        loss_test_history.append(np.mean(minibatch_loss))

        # print the metrics
        print("Train loss: {}, Validation loss: {}".format(loss_train_history[-1], loss_test_history[-1]))
        print("Train AUC: {}, Validation AUC: {}".format(auroc_train_history[-1], auroc_test_history[-1]))
        print("Train Bal.ACC: {}, Validation Bal.ACC: {}".format(balACC_train_history[-1], balACC_test_history[-1]))

        # save model from each epoch====================================================================================
        state = {
            'epochs': _NUM_EPOCHS,
            'experiment_name': _EXPERIMENT_NAME,
            'model_description': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state, f"{_EXPERIMENT_NAME}_Epoch_{epoch}.ckpt")
