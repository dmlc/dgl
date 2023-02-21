import copy
from pathlib import Path

import click

import dgl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data.utils import Subset
from logzero import logger
from modules.dimenet import DimeNet
from modules.dimenet_pp import DimeNetPP
from modules.initializers import GlorotOrthogonal
from qm9 import QM9
from ruamel.yaml import YAML
from sklearn.metrics import mean_absolute_error
from torch.utils.data import DataLoader


def split_dataset(
    dataset, num_train, num_valid, shuffle=False, random_state=None
):
    """Split dataset into training, validation and test set.

    Parameters
    ----------
    dataset
        We assume that ``len(dataset)`` gives the number of datapoints and ``dataset[i]``
        gives the ith datapoint.
    num_train : int
        Number of training datapoints.
    num_valid : int
        Number of validation datapoints.
    shuffle : bool, optional
        By default we perform a consecutive split of the dataset. If True,
        we will first randomly shuffle the dataset.
    random_state : None, int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
        This can be any integer between 0 and 2^32 - 1 inclusive, an array
        (or other sequence) of such integers, or None (the default value).
        If seed is None, then RandomState will try to read data from /dev/urandom
        (or the Windows analogue) if available or seed from the clock otherwise.

    Returns
    -------
    list of length 3
        Subsets for training, validation and test.
    """
    from itertools import accumulate

    num_data = len(dataset)
    assert num_train + num_valid < num_data
    lengths = [num_train, num_valid, num_data - num_train - num_valid]
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]


@torch.no_grad()
def ema(ema_model, model, decay):
    msd = model.state_dict()
    for k, ema_v in ema_model.state_dict().items():
        model_v = msd[k].detach()
        ema_v.copy_(ema_v * decay + (1.0 - decay) * model_v)


def edge_init(edges):
    R_src, R_dst = edges.src["R"], edges.dst["R"]
    dist = torch.sqrt(F.relu(torch.sum((R_src - R_dst) ** 2, -1)))
    # d: bond length, o: bond orientation
    return {"d": dist, "o": R_src - R_dst}


def _collate_fn(batch):
    graphs, line_graphs, labels = map(list, zip(*batch))
    g, l_g = dgl.batch(graphs), dgl.batch(line_graphs)
    labels = torch.tensor(labels, dtype=torch.float32)
    return g, l_g, labels


def train(device, model, opt, loss_fn, train_loader):
    model.train()
    epoch_loss = 0
    num_samples = 0

    for g, l_g, labels in train_loader:
        g = g.to(device)
        l_g = l_g.to(device)
        labels = labels.to(device)
        logits = model(g, l_g)
        loss = loss_fn(logits, labels.view([-1, 1]))
        epoch_loss += loss.data.item() * len(labels)
        num_samples += len(labels)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return epoch_loss / num_samples


@torch.no_grad()
def evaluate(device, model, valid_loader):
    model.eval()
    predictions_all, labels_all = [], []

    for g, l_g, labels in valid_loader:
        g = g.to(device)
        l_g = l_g.to(device)
        logits = model(g, l_g)
        labels_all.extend(labels)
        predictions_all.extend(
            logits.view(
                -1,
            )
            .cpu()
            .numpy()
        )

    return np.array(predictions_all), np.array(labels_all)


@click.command()
@click.option(
    "-m",
    "--model-cnf",
    type=click.Path(exists=True),
    help="Path of model config yaml.",
)
def main(model_cnf):
    yaml = YAML(typ="safe")
    model_cnf = yaml.load(Path(model_cnf))
    model_name, model_params, train_params, pretrain_params = (
        model_cnf["name"],
        model_cnf["model"],
        model_cnf["train"],
        model_cnf["pretrain"],
    )
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model params: {model_params}")
    logger.info(f"Train params: {train_params}")

    if model_params["targets"] in ["mu", "homo", "lumo", "gap", "zpve"]:
        model_params["output_init"] = nn.init.zeros_
    else:
        # 'GlorotOrthogonal' for alpha, R2, U0, U, H, G, and Cv
        model_params["output_init"] = GlorotOrthogonal

    logger.info("Loading Data Set")
    dataset = QM9(label_keys=model_params["targets"], edge_funcs=[edge_init])

    # data split
    train_data, valid_data, test_data = split_dataset(
        dataset,
        num_train=train_params["num_train"],
        num_valid=train_params["num_valid"],
        shuffle=True,
        random_state=train_params["data_seed"],
    )
    logger.info(f"Size of Training Set: {len(train_data)}")
    logger.info(f"Size of Validation Set: {len(valid_data)}")
    logger.info(f"Size of Test Set: {len(test_data)}")

    # data loader
    train_loader = DataLoader(
        train_data,
        batch_size=train_params["batch_size"],
        shuffle=True,
        collate_fn=_collate_fn,
        num_workers=train_params["num_workers"],
    )

    valid_loader = DataLoader(
        valid_data,
        batch_size=train_params["batch_size"],
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=train_params["num_workers"],
    )

    test_loader = DataLoader(
        test_data,
        batch_size=train_params["batch_size"],
        shuffle=False,
        collate_fn=_collate_fn,
        num_workers=train_params["num_workers"],
    )

    # check cuda
    gpu = train_params["gpu"]
    device = f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu"

    # model initialization
    logger.info("Loading Model")
    if model_name == "dimenet":
        model = DimeNet(
            emb_size=model_params["emb_size"],
            num_blocks=model_params["num_blocks"],
            num_bilinear=model_params["num_bilinear"],
            num_spherical=model_params["num_spherical"],
            num_radial=model_params["num_radial"],
            cutoff=model_params["cutoff"],
            envelope_exponent=model_params["envelope_exponent"],
            num_before_skip=model_params["num_before_skip"],
            num_after_skip=model_params["num_after_skip"],
            num_dense_output=model_params["num_dense_output"],
            num_targets=len(model_params["targets"]),
            output_init=model_params["output_init"],
        ).to(device)
    elif model_name == "dimenet++":
        model = DimeNetPP(
            emb_size=model_params["emb_size"],
            out_emb_size=model_params["out_emb_size"],
            int_emb_size=model_params["int_emb_size"],
            basis_emb_size=model_params["basis_emb_size"],
            num_blocks=model_params["num_blocks"],
            num_spherical=model_params["num_spherical"],
            num_radial=model_params["num_radial"],
            cutoff=model_params["cutoff"],
            envelope_exponent=model_params["envelope_exponent"],
            num_before_skip=model_params["num_before_skip"],
            num_after_skip=model_params["num_after_skip"],
            num_dense_output=model_params["num_dense_output"],
            num_targets=len(model_params["targets"]),
            extensive=model_params["extensive"],
            output_init=model_params["output_init"],
        ).to(device)
    else:
        raise ValueError(f"Invalid Model Name {model_name}")

    if pretrain_params["flag"]:
        torch_path = pretrain_params["path"]
        target = model_params["targets"][0]
        model.load_state_dict(torch.load(f"{torch_path}/{target}.pt"))

        logger.info("Testing with Pretrained model")
        predictions, labels = evaluate(device, model, test_loader)
        test_mae = mean_absolute_error(labels, predictions)
        logger.info(f"Test MAE {test_mae:.4f}")

        return
    # define loss function and optimization
    loss_fn = nn.L1Loss()
    opt = optim.Adam(
        model.parameters(),
        lr=train_params["lr"],
        weight_decay=train_params["weight_decay"],
        amsgrad=True,
    )
    scheduler = optim.lr_scheduler.StepLR(
        opt, train_params["step_size"], gamma=train_params["gamma"]
    )

    # model training
    best_mae = 1e9
    no_improvement = 0

    # EMA for valid and test
    logger.info("EMA Init")
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    best_model = copy.deepcopy(ema_model)

    logger.info("Training")
    for i in range(train_params["epochs"]):
        train_loss = train(device, model, opt, loss_fn, train_loader)
        ema(ema_model, model, train_params["ema_decay"])
        if i % train_params["interval"] == 0:
            predictions, labels = evaluate(device, ema_model, valid_loader)

            valid_mae = mean_absolute_error(labels, predictions)
            logger.info(
                f"Epoch {i} | Train Loss {train_loss:.4f} | Val MAE {valid_mae:.4f}"
            )

            if valid_mae > best_mae:
                no_improvement += 1
                if no_improvement == train_params["early_stopping"]:
                    logger.info("Early stop.")
                    break
            else:
                no_improvement = 0
                best_mae = valid_mae
                best_model = copy.deepcopy(ema_model)
        else:
            logger.info(f"Epoch {i} | Train Loss {train_loss:.4f}")

        scheduler.step()

    logger.info("Testing")
    predictions, labels = evaluate(device, best_model, test_loader)
    test_mae = mean_absolute_error(labels, predictions)
    logger.info("Test MAE {:.4f}".format(test_mae))


if __name__ == "__main__":
    main()
