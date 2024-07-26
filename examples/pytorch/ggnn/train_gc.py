"""
Training and testing for graph classification tasks in bAbI
"""

import argparse

import numpy as np
import torch
from data_utils import get_babi_dataloaders
from ggnn_gc import GraphClsGGNN
from torch.optim import Adam


def main(args):
    out_feats = {18: 3}
    n_etypes = {18: 2}

    train_dataloader, dev_dataloader, test_dataloaders = get_babi_dataloaders(
        batch_size=args.batch_size,
        train_size=args.train_num,
        task_id=args.task_id,
        q_type=args.question_id,
    )

    model = GraphClsGGNN(
        annotation_size=2,
        out_feats=out_feats[args.task_id],
        n_steps=5,
        n_etypes=n_etypes[args.task_id],
        num_cls=2,
    )
    opt = Adam(model.parameters(), lr=args.lr)

    print(f"Task {args.task_id}, question_id {args.question_id}")

    print(f"Training set size: {len(train_dataloader.dataset)}")
    print(f"Dev set size: {len(dev_dataloader.dataset)}")

    # training and dev stage
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            g, labels = batch
            loss, _ = model(g, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, batch {i} loss: {loss.data}")

        if epoch % 20 != 0:
            continue
        dev_preds = []
        dev_labels = []
        model.eval()
        for g, labels in dev_dataloader:
            with torch.no_grad():
                preds = model(g)
                preds = preds.data.numpy().tolist()
                labels = labels.data.numpy().tolist()
                dev_preds += preds
                dev_labels += labels
        acc = np.equal(dev_labels, dev_preds).astype(float).tolist()
        acc = sum(acc) / len(acc)
        print(f"Epoch {epoch}, Dev acc {acc}")

    # test stage
    for i, dataloader in enumerate(test_dataloaders):
        print(f"Test set {i} size: {len(dataloader.dataset)}")

    test_acc_list = []
    for dataloader in test_dataloaders:
        test_preds = []
        test_labels = []
        model.eval()
        for g, labels in dataloader:
            with torch.no_grad():
                preds = model(g)
                preds = preds.data.numpy().tolist()
                labels = labels.data.numpy().tolist()
                test_preds += preds
                test_labels += labels
        acc = np.equal(test_labels, test_preds).astype(float).tolist()
        acc = sum(acc) / len(acc)
        test_acc_list.append(acc)

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    print(
        f"Mean of accuracy in 10 test datasets: {test_acc_mean}, std: {test_acc_std}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gated Graph Neural Networks for graph classification tasks in bAbI"
    )
    parser.add_argument(
        "--task_id", type=int, default=18, help="task id from 1 to 20"
    )
    parser.add_argument(
        "--question_id", type=int, default=0, help="question id for each task"
    )
    parser.add_argument(
        "--train_num", type=int, default=950, help="Number of training examples"
    )
    parser.add_argument("--batch_size", type=int, default=50, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )

    args = parser.parse_args()

    main(args)
