import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from dgl.dataloading import GraphDataLoader
from EEGGraphDataset import EEGGraphDataset
from joblib import dump, load
from sklearn import preprocessing
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler


def _load_memory_mapped_array(file_name):
    # Due to a legacy problem related to memory alignment in joblib [1], the
    # data provided in the example may not be byte-aligned. This can be risky
    # when loading with mmap_mode. To fix the issue, load and re-dump the data.
    # [1] https://joblib.readthedocs.io/en/latest/developing.html#release-1-2-0
    dump(load(file_name), file_name)
    return load(file_name, mmap_mode="r")


if __name__ == "__main__":
    # argparse commandline args
    parser = argparse.ArgumentParser(
        description="Execute training pipeline on a given train/val subjects"
    )
    parser.add_argument(
        "--num_feats",
        type=int,
        default=6,
        help="Number of features per node for the graph",
    )
    parser.add_argument(
        "--num_nodes", type=int, default=8, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of epochs used to train",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="index of GPU device that should be used for this run, defaults to 0.",
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=40,
        help="Number of epochs used to train",
    )
    parser.add_argument(
        "--exp_name", type=str, default="default", help="Name for the test."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch Size. Default is 512.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="shallow",
        help="type shallow to use shallow_EEGGraphDataset; "
        "type deep to use deep_EEGGraphDataset. Default is shallow",
    )
    args = parser.parse_args()

    # choose model
    if args.model == "shallow":
        from shallow_EEGGraphConvNet import EEGGraphConvNet

    if args.model == "deep":
        from deep_EEGGraphConvNet import EEGGraphConvNet

    # set the random seed so that we can reproduce the results
    np.random.seed(42)
    torch.manual_seed(42)

    # use GPU when available
    _GPU_IDX = args.gpu_idx
    _DEVICE = torch.device(
        f"cuda:{_GPU_IDX}" if torch.cuda.is_available() else "cpu"
    )
    torch.cuda.set_device(_DEVICE)
    print(f" Using device: {_DEVICE} {torch.cuda.get_device_name(_DEVICE)}")

    # load patient level indices
    _DATASET_INDEX = pd.read_csv("master_metadata_index.csv", low_memory=False)
    all_subjects = _DATASET_INDEX["patient_ID"].astype("str").unique()
    print(f"Subject list fetched! Total subjects are {len(all_subjects)}.")

    # retrieve inputs
    num_nodes = args.num_nodes
    _NUM_EPOCHS = args.num_epochs
    _EXPERIMENT_NAME = args.exp_name
    _BATCH_SIZE = args.batch_size
    num_feats = args.num_feats
    num_workers = args.num_workers

    # set up input and targets from files
    x = _load_memory_mapped_array(f"psd_features_data_X")
    y = _load_memory_mapped_array(f"labels_y")

    # normalize psd features data
    normd_x = []
    for i in range(len(y)):
        arr = x[i, :]
        arr = arr.reshape(1, -1)
        arr2 = preprocessing.normalize(arr)
        arr2 = arr2.reshape(48)
        normd_x.append(arr2)

    norm = np.array(normd_x)
    x = norm.reshape(len(y), 48)
    # map 0/1 to diseased/healthy
    label_mapping, y = np.unique(y, return_inverse=True)
    print(f"Unique labels 0/1 mapping: {label_mapping}")

    # split the dataset to train and test. The ratio of test is 0.3.
    train_and_val_subjects, heldout_subjects = train_test_split(
        all_subjects, test_size=0.3, random_state=42
    )

    # split the dataset using patient indices
    train_window_indices = _DATASET_INDEX.index[
        _DATASET_INDEX["patient_ID"].astype("str").isin(train_and_val_subjects)
    ].tolist()
    heldout_test_window_indices = _DATASET_INDEX.index[
        _DATASET_INDEX["patient_ID"].astype("str").isin(heldout_subjects)
    ].tolist()

    # define model, optimizer, scheduler
    model = EEGGraphConvNet(num_feats)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[i * 10 for i in range(1, 26)], gamma=0.1
    )

    model = model.to(_DEVICE).double()
    num_trainable_params = np.sum(
        [
            np.prod(p.size()) if p.requires_grad else 0
            for p in model.parameters()
        ]
    )

    # Dataloader========================================================================================================

    # use WeightedRandomSampler to balance the training dataset

    labels_unique, counts = np.unique(y, return_counts=True)

    class_weights = np.array([1.0 / x for x in counts])
    # provide weights for samples in the training set only
    sample_weights = class_weights[y[train_window_indices]]
    # sampler needs to come up with training set size number of samples
    weighted_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_window_indices),
        replacement=True,
    )

    # train data loader
    train_dataset = EEGGraphDataset(
        x=x, y=y, num_nodes=num_nodes, indices=train_window_indices
    )

    train_loader = GraphDataLoader(
        dataset=train_dataset,
        batch_size=_BATCH_SIZE,
        sampler=weighted_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )

    # this loader is used without weighted sampling, to evaluate metrics on full training set after each epoch
    train_metrics_loader = GraphDataLoader(
        dataset=train_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # test data loader
    test_dataset = EEGGraphDataset(
        x=x, y=y, num_nodes=num_nodes, indices=heldout_test_window_indices
    )

    test_loader = GraphDataLoader(
        dataset=test_dataset,
        batch_size=_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

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
        y_probs_train = (
            nn.functional.softmax(y_probs_train, dim=1).cpu().numpy()
        )
        y_true_train = np.array(y_true_train)

        # evaluate model after each epoch for validation data ==============================================================
        y_probs_test = torch.empty(0, 2).to(_DEVICE)
        y_true_test, minibatch_loss, y_pred_test = [], [], []

        for i, batch in enumerate(test_loader):
            g, dataset_idx, y = batch
            g_batch = g.to(device=_DEVICE, non_blocking=True)
            y_batch = y.to(device=_DEVICE, non_blocking=True)

            # forward pass
            outputs = model(g_batch)
            _, predicted = torch.max(outputs.data, 1)
            y_pred_test += predicted.cpu().numpy().tolist()

            loss = loss_function(outputs, y_batch)
            minibatch_loss.append(loss.item())
            y_probs_test = torch.cat((y_probs_test, outputs.data), 0)
            y_true_test += y_batch.cpu().numpy().tolist()

        # returning prob distribution over target classes, take softmax over the 1st dimension
        y_probs_test = (
            torch.nn.functional.softmax(y_probs_test, dim=1).cpu().numpy()
        )
        y_true_test = np.array(y_true_test)

        # record training auroc and testing auroc
        auroc_train_history.append(
            roc_auc_score(y_true_train, y_probs_train[:, 1])
        )
        auroc_test_history.append(
            roc_auc_score(y_true_test, y_probs_test[:, 1])
        )

        # record training balanced accuracy and testing balanced accuracy
        balACC_train_history.append(
            balanced_accuracy_score(y_true_train, y_pred_train)
        )
        balACC_test_history.append(
            balanced_accuracy_score(y_true_test, y_pred_test)
        )

        # LOSS - epoch loss is defined as mean of minibatch losses within epoch
        loss_train_history.append(np.mean(train_loss))
        loss_test_history.append(np.mean(minibatch_loss))

        # print the metrics
        print(
            "Train loss: {}, test loss: {}".format(
                loss_train_history[-1], loss_test_history[-1]
            )
        )
        print(
            "Train AUC: {}, test AUC: {}".format(
                auroc_train_history[-1], auroc_test_history[-1]
            )
        )
        print(
            "Train Bal.ACC: {}, test Bal.ACC: {}".format(
                balACC_train_history[-1], balACC_test_history[-1]
            )
        )

        # save model from each epoch====================================================================================
        state = {
            "epochs": _NUM_EPOCHS,
            "experiment_name": _EXPERIMENT_NAME,
            "model_description": str(model),
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, f"{_EXPERIMENT_NAME}_Epoch_{epoch}.ckpt")
