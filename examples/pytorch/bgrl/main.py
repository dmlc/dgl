import copy
import os
import warnings

import dgl

import numpy as np
import torch
from eval_function import (
    fit_logistic_regression,
    fit_logistic_regression_preset_splits,
    fit_ppi_linear,
)
from model import (
    BGRL,
    compute_representations,
    GCN,
    GraphSAGE_GCN,
    MLP_Predictor,
)
from torch.nn.functional import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm
from utils import CosineDecayScheduler, get_dataset, get_graph_drop_transform

warnings.filterwarnings("ignore")


def train(
    step,
    model,
    optimizer,
    lr_scheduler,
    mm_scheduler,
    transform_1,
    transform_2,
    data,
    args,
):
    model.train()

    # update learning rate
    lr = lr_scheduler.get(step)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # update momentum
    mm = 1 - mm_scheduler.get(step)

    # forward
    optimizer.zero_grad()

    x1, x2 = transform_1(data), transform_2(data)

    if args.dataset != "ppi":
        x1, x2 = dgl.add_self_loop(x1), dgl.add_self_loop(x2)

    q1, y2 = model(x1, x2)
    q2, y1 = model(x2, x1)

    loss = (
        2
        - cosine_similarity(q1, y2.detach(), dim=-1).mean()
        - cosine_similarity(q2, y1.detach(), dim=-1).mean()
    )
    loss.backward()

    # update online network
    optimizer.step()
    # update target network
    model.update_target_network(mm)

    return loss.item()


def eval(model, dataset, device, args, train_data, val_data, test_data):
    # make temporary copy of encoder
    tmp_encoder = copy.deepcopy(model.online_encoder).eval()
    val_scores = None

    if args.dataset == "ppi":
        train_data = compute_representations(tmp_encoder, train_data, device)
        val_data = compute_representations(tmp_encoder, val_data, device)
        test_data = compute_representations(tmp_encoder, test_data, device)
        num_classes = train_data[1].shape[1]
        val_scores, test_scores = fit_ppi_linear(
            num_classes,
            train_data,
            val_data,
            test_data,
            device,
            args.num_eval_splits,
        )
    elif args.dataset != "wiki_cs":
        representations, labels = compute_representations(
            tmp_encoder, dataset, device
        )
        test_scores = fit_logistic_regression(
            representations.cpu().numpy(),
            labels.cpu().numpy(),
            data_random_seed=args.data_seed,
            repeat=args.num_eval_splits,
        )
    else:
        g = dataset[0]
        train_mask = g.ndata["train_mask"]
        val_mask = g.ndata["val_mask"]
        test_mask = g.ndata["test_mask"]
        representations, labels = compute_representations(
            tmp_encoder, dataset, device
        )
        test_scores = fit_logistic_regression_preset_splits(
            representations.cpu().numpy(),
            labels.cpu().numpy(),
            train_mask,
            val_mask,
            test_mask,
        )

    return val_scores, test_scores


def main(args):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print("Using device:", device)

    dataset, train_data, val_data, test_data = get_dataset(args.dataset)

    g = dataset[0]
    g = g.to(device)

    input_size, representation_size = (
        g.ndata["feat"].size(1),
        args.graph_encoder_layer[-1],
    )

    # prepare transforms
    transform_1 = get_graph_drop_transform(
        drop_edge_p=args.drop_edge_p[0], feat_mask_p=args.feat_mask_p[0]
    )
    transform_2 = get_graph_drop_transform(
        drop_edge_p=args.drop_edge_p[1], feat_mask_p=args.feat_mask_p[1]
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(
        args.lr, args.lr_warmup_epochs, args.epochs
    )
    mm_scheduler = CosineDecayScheduler(1 - args.mm, 0, args.epochs)

    # build networks
    if args.dataset == "ppi":
        encoder = GraphSAGE_GCN([input_size] + args.graph_encoder_layer)
    else:
        encoder = GCN([input_size] + args.graph_encoder_layer)
    predictor = MLP_Predictor(
        representation_size,
        representation_size,
        hidden_size=args.predictor_hidden_size,
    )
    model = BGRL(encoder, predictor).to(device)

    # optimizer
    optimizer = AdamW(
        model.trainable_parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # train
    for epoch in tqdm(range(1, args.epochs + 1), desc="  - (Training)  "):
        train(
            epoch - 1,
            model,
            optimizer,
            lr_scheduler,
            mm_scheduler,
            transform_1,
            transform_2,
            g,
            args,
        )
        if epoch % args.eval_epochs == 0:
            val_scores, test_scores = eval(
                model, dataset, device, args, train_data, val_data, test_data
            )
            if args.dataset == "ppi":
                print(
                    "Epoch: {:04d} | Best Val F1: {:.4f} | Test F1: {:.4f}".format(
                        epoch, np.mean(val_scores), np.mean(test_scores)
                    )
                )
            else:
                print(
                    "Epoch: {:04d} | Test Accuracy: {:.4f}".format(
                        epoch, np.mean(test_scores)
                    )
                )

    # save encoder weights
    if not os.path.isdir(args.weights_dir):
        os.mkdir(args.weights_dir)
    torch.save(
        {"model": model.online_encoder.state_dict()},
        os.path.join(args.weights_dir, "bgrl-{}.pt".format(args.dataset)),
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    # Dataset options.
    parser.add_argument(
        "--dataset",
        type=str,
        default="amazon_photos",
        choices=[
            "coauthor_cs",
            "coauthor_physics",
            "amazon_photos",
            "amazon_computers",
            "wiki_cs",
            "ppi",
        ],
    )

    # Model options.
    parser.add_argument(
        "--graph_encoder_layer", type=int, nargs="+", default=[256, 128]
    )
    parser.add_argument("--predictor_hidden_size", type=int, default=512)

    # Training options.
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--mm", type=float, default=0.99)
    parser.add_argument("--lr_warmup_epochs", type=int, default=1000)
    parser.add_argument("--weights_dir", type=str, default="../weights")

    # Augmentations options.
    parser.add_argument(
        "--drop_edge_p", type=float, nargs="+", default=[0.0, 0.0]
    )
    parser.add_argument(
        "--feat_mask_p", type=float, nargs="+", default=[0.0, 0.0]
    )

    # Evaluation options.
    parser.add_argument("--eval_epochs", type=int, default=250)
    parser.add_argument("--num_eval_splits", type=int, default=20)
    parser.add_argument("--data_seed", type=int, default=1)

    # Experiment options.
    parser.add_argument("--num_experiments", type=int, default=20)

    args = parser.parse_args()

    main(args)
