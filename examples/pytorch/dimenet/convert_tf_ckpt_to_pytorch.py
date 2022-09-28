import os
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
from logzero import logger
from modules.dimenet_pp import DimeNetPP
from modules.initializers import GlorotOrthogonal
from ruamel.yaml import YAML


@click.command()
@click.option(
    "-m",
    "--model-cnf",
    type=click.Path(exists=True),
    help="Path of model config yaml.",
)
@click.option(
    "-c",
    "--convert-cnf",
    type=click.Path(exists=True),
    help="Path of convert config yaml.",
)
def main(model_cnf, convert_cnf):
    yaml = YAML(typ="safe")
    model_cnf = yaml.load(Path(model_cnf))
    convert_cnf = yaml.load(Path(convert_cnf))
    model_name, model_params, _ = (
        model_cnf["name"],
        model_cnf["model"],
        model_cnf["train"],
    )
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model params: {model_params}")

    if model_params["targets"] in ["mu", "homo", "lumo", "gap", "zpve"]:
        model_params["output_init"] = nn.init.zeros_
    else:
        # 'GlorotOrthogonal' for alpha, R2, U0, U, H, G, and Cv
        model_params["output_init"] = GlorotOrthogonal

    # model initialization
    logger.info("Loading Model")
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
    )
    logger.info(model.state_dict())
    tf_path, torch_path = (
        convert_cnf["tf"]["ckpt_path"],
        convert_cnf["torch"]["dump_path"],
    )
    init_vars = tf.train.list_variables(tf_path)
    tf_vars_dict = {}

    # 147 keys
    for name, shape in init_vars:
        if name == "_CHECKPOINTABLE_OBJECT_GRAPH":
            continue
        array = tf.train.load_variable(tf_path, name)
        logger.info(f"Loading TF weight {name} with shape {shape}")
        tf_vars_dict[name] = array

    for name, array in tf_vars_dict.items():
        name = name.split("/")[:-2]
        pointer = model

        for m_name in name:
            if m_name == "kernel":
                pointer = getattr(pointer, "weight")
            elif m_name == "int_blocks":
                pointer = getattr(pointer, "interaction_blocks")
            elif m_name == "embeddings":
                pointer = getattr(pointer, "embedding")
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, m_name)
        if name[-1] == "kernel":
            array = np.transpose(array)
        assert array.shape == pointer.shape
        logger.info(f"Initialize PyTorch weight {name}")
        pointer.data = torch.from_numpy(array)

    logger.info(f"Save PyTorch model to {torch_path}")
    if not os.path.exists(torch_path):
        os.makedirs(torch_path)
    target = model_params["targets"][0]
    torch.save(model.state_dict(), f"{torch_path}/{target}.pt")
    logger.info(model.state_dict())


if __name__ == "__main__":
    main()
