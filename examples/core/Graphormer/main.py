"""
This script finetunes and tests a Graphormer model (pretrained on PCQM4Mv2)
for graph classification on ogbg-molhiv dataset.

Paper: [Do Transformers Really Perform Bad for Graph Representation?]
(https://arxiv.org/abs/2106.05234)

This flowchart describes the main functional sequence of the provided example.
main
│
└───> train_val_pipeline
      │
      ├───> Load and preprocess dataset
      │
      ├───> Download pretrained model
      │
      ├───> train_epoch
      │     │
      │     └───> Graphormer.forward
      │
      └───> evaluate_network
            │
            └───> Graphormer.inference
"""
import argparse
import random

import torch as th
import torch.nn as nn
from accelerate import Accelerator
from dataset import MolHIVDataset
from dgl.data import download
from dgl.dataloading import GraphDataLoader
from model import Graphormer
from ogb.graphproppred import Evaluator
from transformers.optimization import (
    AdamW,
    get_polynomial_decay_schedule_with_warmup,
)

# Instantiate an accelerator object to support distributed
# training and inference.
accelerator = Accelerator()


def train_epoch(model, optimizer, data_loader, lr_scheduler):
    model.train()
    epoch_loss = 0
    list_scores = []
    list_labels = []
    loss_fn = nn.BCEWithLogitsLoss()
    for (
        batch_labels,
        attn_mask,
        node_feat,
        in_degree,
        out_degree,
        path_data,
        dist,
    ) in data_loader:
        optimizer.zero_grad()
        device = accelerator.device

        batch_scores = model(
            node_feat.to(device),
            in_degree.to(device),
            out_degree.to(device),
            path_data.to(device),
            dist.to(device),
            attn_mask=attn_mask,
        )

        loss = loss_fn(batch_scores, batch_labels.float())

        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        epoch_loss += loss.item()
        list_scores.append(batch_scores)
        list_labels.append(batch_labels)

        # Release GPU memory.
        del (
            batch_labels,
            batch_scores,
            loss,
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
        )
        th.cuda.empty_cache()

    epoch_loss /= len(data_loader)

    evaluator = Evaluator(name="ogbg-molhiv")
    epoch_auc = evaluator.eval(
        {"y_pred": th.cat(list_scores), "y_true": th.cat(list_labels)}
    )["rocauc"]

    return epoch_loss, epoch_auc


def evaluate_network(model, data_loader):
    model.eval()
    epoch_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    with th.no_grad():
        list_scores = []
        list_labels = []
        for (
            batch_labels,
            attn_mask,
            node_feat,
            in_degree,
            out_degree,
            path_data,
            dist,
        ) in data_loader:
            device = accelerator.device

            batch_scores = model(
                node_feat.to(device),
                in_degree.to(device),
                out_degree.to(device),
                path_data.to(device),
                dist.to(device),
                attn_mask=attn_mask,
            )

            # Gather all predictions and targets.
            all_predictions, all_targets = accelerator.gather_for_metrics(
                (batch_scores, batch_labels)
            )
            loss = loss_fn(all_predictions, all_targets.float())

            epoch_loss += loss.item()
            list_scores.append(all_predictions)
            list_labels.append(all_targets)

        epoch_loss /= len(data_loader)

        evaluator = Evaluator(name="ogbg-molhiv")
        epoch_auc = evaluator.eval(
            {"y_pred": th.cat(list_scores), "y_true": th.cat(list_labels)}
        )["rocauc"]

    return epoch_loss, epoch_auc


def train_val_pipeline(params):
    dataset = MolHIVDataset()

    accelerator.print(
        f"train, test, val sizes: {len(dataset.train)}, "
        f"{len(dataset.test)}, {len(dataset.val)}."
    )
    accelerator.print("Finished loading.")

    train_loader = GraphDataLoader(
        dataset.train,
        batch_size=params.batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )
    val_loader = GraphDataLoader(
        dataset.val,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )
    test_loader = GraphDataLoader(
        dataset.test,
        batch_size=params.batch_size,
        shuffle=False,
        collate_fn=dataset.collate,
        pin_memory=True,
        num_workers=16,
    )

    # Load pre-trained model.
    download(url="https://data.dgl.ai/pre_trained/graphormer_pcqm.pth")
    model = Graphormer()
    state_dict = th.load("graphormer_pcqm.pth")
    model.load_state_dict(state_dict)

    model.reset_output_layer_parameters()
    num_epochs = 16
    total_updates = 33000 * num_epochs / params.batch_size
    # Use warmup schedule to avoid overfitting at the very beginning
    # of training, the ratio 0.16 is the same as the paper.
    warmup_updates = total_updates * 0.16

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8, weight_decay=0)
    lr_scheduler = get_polynomial_decay_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_updates,
        num_training_steps=total_updates,
        lr_end=1e-9,
        power=1.0,
    )

    epoch_train_AUCs, epoch_val_AUCs, epoch_test_AUCs = [], [], []

    # Pass all objects relevant to training to the prepare() method as required
    # by Accelerate.
    (
        model,
        optimizer,
        train_loader,
        val_loader,
        test_loader,
        lr_scheduler,
    ) = accelerator.prepare(
        model, optimizer, train_loader, val_loader, test_loader, lr_scheduler
    )

    for epoch in range(num_epochs):
        epoch_train_loss, epoch_train_auc = train_epoch(
            model, optimizer, train_loader, lr_scheduler
        )
        epoch_val_loss, epoch_val_auc = evaluate_network(model, val_loader)
        epoch_test_loss, epoch_test_auc = evaluate_network(model, test_loader)

        epoch_train_AUCs.append(epoch_train_auc)
        epoch_val_AUCs.append(epoch_val_auc)
        epoch_test_AUCs.append(epoch_test_auc)

        accelerator.print(
            f"Epoch={epoch + 1} | train_AUC={epoch_train_auc:.3f} | "
            f"val_AUC={epoch_val_auc:.3f} | test_AUC={epoch_test_auc:.3f}"
        )

    # Return test and train AUCs with best val AUC.
    index = epoch_val_AUCs.index(max(epoch_val_AUCs))
    val_auc = epoch_val_AUCs[index]
    train_auc = epoch_train_AUCs[index]
    test_auc = epoch_test_AUCs[index]

    accelerator.print("Test ROCAUC: {:.4f}".format(test_auc))
    accelerator.print("Val ROCAUC: {:.4f}".format(val_auc))
    accelerator.print("Train ROCAUC: {:.4f}".format(train_auc))
    accelerator.print("Best epoch index: {:.4f}".format(index))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        default=1,
        type=int,
        help="Please give a value for random seed",
    )
    parser.add_argument(
        "--batch_size",
        default=16,
        type=int,
        help="Please give a value for batch_size",
    )
    args = parser.parse_args()

    # Set manual seed to bind the order of training data to the random seed.
    random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(args.seed)

    train_val_pipeline(args)
