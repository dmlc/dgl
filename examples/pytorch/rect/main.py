import torch
import torch.nn as nn
from classify import evaluate_embeds
from label_utils import (
    get_labeled_nodes_label_attribute,
    remove_unseen_classes_from_training,
)
from model import GCN, RECT_L
from utils import load_data, process_classids, svd_feature


def main(args):
    g, features, labels, train_mask, test_mask, n_classes, cuda = load_data(
        args
    )
    # adopt any number of classes as the unseen classes (the first three classes by default)
    removed_class = args.removed_class
    if len(removed_class) > n_classes:
        raise ValueError(
            "unseen number is greater than the number of classes: {}".format(
                len(removed_class)
            )
        )
    for i in removed_class:
        if i not in labels:
            raise ValueError("class out of bounds: {}".format(i))

    # remove these unseen classes from the training set, to construct the zero-shot label setting
    train_mask_zs = remove_unseen_classes_from_training(
        train_mask=train_mask, labels=labels, removed_class=removed_class
    )
    print(
        "after removing the unseen classes, seen class labeled node num:",
        sum(train_mask_zs).item(),
    )

    if args.model_opt == "RECT-L":
        model = RECT_L(
            g=g,
            in_feats=args.n_hidden,
            n_hidden=args.n_hidden,
            activation=nn.PReLU(),
        )

        if cuda:
            model.cuda()
        features = svd_feature(features=features, d=args.n_hidden)
        attribute_labels = get_labeled_nodes_label_attribute(
            train_mask_zs=train_mask_zs,
            labels=labels,
            features=features,
            cuda=cuda,
        )
        loss_fcn = nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(args.n_epochs):
            model.train()
            optimizer.zero_grad()
            logits = model(features)
            loss_train = loss_fcn(attribute_labels, logits[train_mask_zs])
            print(
                "Epoch {:d} | Train Loss {:.5f}".format(
                    epoch + 1, loss_train.item()
                )
            )
            loss_train.backward()
            optimizer.step()
        model.eval()
        embeds = model.embed(features)

    elif args.model_opt == "GCN":
        model = GCN(
            g=g,
            in_feats=features.shape[1],
            n_hidden=args.n_hidden,
            n_classes=n_classes - len(removed_class),
            activation=nn.PReLU(),
            dropout=args.dropout,
        )

        if cuda:
            model.cuda()
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        for epoch in range(args.n_epochs):
            model.train()
            logits = model(features)
            labels_train = process_classids(labels_temp=labels[train_mask_zs])
            loss_train = loss_fcn(logits[train_mask_zs], labels_train)
            optimizer.zero_grad()
            print(
                "Epoch {:d} | Train Loss {:.5f}".format(
                    epoch + 1, loss_train.item()
                )
            )
            loss_train.backward()
            optimizer.step()
        model.eval()
        embeds = model.embed(features)

    elif args.model_opt == "NodeFeats":
        embeds = svd_feature(features)

    # evaluate the quality of embedding results with the original balanced labels, to assess the model performance (as suggested in the paper)
    res = evaluate_embeds(
        features=embeds,
        labels=labels,
        train_mask=train_mask,
        test_mask=test_mask,
        n_classes=n_classes,
        cuda=cuda,
    )
    print("Test Accuracy of {:s}: {:.4f}".format(args.model_opt, res))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MODEL")
    parser.add_argument(
        "--model-opt",
        type=str,
        default="RECT-L",
        choices=["RECT-L", "GCN", "NodeFeats"],
        help="model option",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        choices=["cora", "citeseer"],
        help="dataset",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0, help="dropout probability"
    )
    parser.add_argument("--gpu", type=int, default=0, help="gpu")
    parser.add_argument(
        "--removed-class",
        type=int,
        nargs="*",
        default=[0, 1, 2],
        help="remove the unseen classes",
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="number of training epochs"
    )
    parser.add_argument(
        "--n-hidden", type=int, default=200, help="number of hidden gcn units"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=5e-4, help="Weight for L2 loss"
    )
    args = parser.parse_args()

    main(args)
