"""
Training and testing for sequence output tasks in bAbI.
Here we take task 19 'Path Finding' as an example
"""

import argparse

import numpy as np
import torch
from data_utils import get_babi_dataloaders
from ggsnn import GGSNN
from torch.optim import Adam


def main(args):
    out_feats = {19: 6}
    n_etypes = {19: 4}

    train_dataloader, dev_dataloader, test_dataloaders = get_babi_dataloaders(
        batch_size=args.batch_size,
        train_size=args.train_num,
        task_id=args.task_id,
        q_type=-1,
    )

    model = GGSNN(
        annotation_size=2,
        out_feats=out_feats[args.task_id],
        n_steps=5,
        n_etypes=n_etypes[args.task_id],
        max_seq_length=2,
        num_cls=5,
    )
    opt = Adam(model.parameters(), lr=args.lr)

    print(f"Task {args.task_id}")

    print(f"Training set size: {len(train_dataloader.dataset)}")
    print(f"Dev set size: {len(dev_dataloader.dataset)}")

    # training and dev stage
    for epoch in range(args.epochs):
        model.train()
        for i, batch in enumerate(train_dataloader):
            g, ground_truths, seq_lengths = batch
            loss, _ = model(g, seq_lengths, ground_truths)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, batch {i} loss: {loss.data}")

        if epoch % 20 != 0:
            continue
        dev_res = []
        model.eval()
        for g, ground_truths, seq_lengths in dev_dataloader:
            with torch.no_grad():
                preds = model(g, seq_lengths)
                preds = preds.data.numpy().tolist()
                ground_truths = ground_truths.data.numpy().tolist()
                for i, p in enumerate(preds):
                    if p == ground_truths[i]:
                        dev_res.append(1.0)
                    else:
                        dev_res.append(0.0)
        acc = sum(dev_res) / len(dev_res)
        print(f"Epoch {epoch}, Dev acc {acc}")

    # test stage
    for i, dataloader in enumerate(test_dataloaders):
        print(f"Test set {i} size: {len(dataloader.dataset)}")

    test_acc_list = []
    for dataloader in test_dataloaders:
        test_res = []
        model.eval()
        for g, ground_truths, seq_lengths in dataloader:
            with torch.no_grad():
                preds = model(g, seq_lengths)
                preds = preds.data.numpy().tolist()
                ground_truths = ground_truths.data.numpy().tolist()
                for i, p in enumerate(preds):
                    if p == ground_truths[i]:
                        test_res.append(1.0)
                    else:
                        test_res.append(0.0)
        acc = sum(test_res) / len(test_res)
        test_acc_list.append(acc)

    test_acc_mean = np.mean(test_acc_list)
    test_acc_std = np.std(test_acc_list)

    print(
        f"Mean of accuracy in 10 test datasets: {test_acc_mean}, std: {test_acc_std}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Gated Graph Sequence Neural Networks for sequential output tasks in "
        "bAbI"
    )
    parser.add_argument(
        "--task_id", type=int, default=19, help="task id from 1 to 20"
    )
    parser.add_argument(
        "--train_num", type=int, default=250, help="Number of training examples"
    )
    parser.add_argument("--batch_size", type=int, default=10, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--epochs", type=int, default=200, help="number of training epochs"
    )

    args = parser.parse_args()

    main(args)
