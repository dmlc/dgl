import copy
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import ruamel.yaml

import typer
import yaml
from jinja2 import ext, Template
from pydantic import BaseModel, Field
from ruamel.yaml.comments import CommentedMap
from typing_extensions import Literal

from ...utils.base_model import DeviceEnum, EarlyStopConfig, extract_name
from ...utils.factory import (
    DataFactory,
    NodeModelFactory,
    PipelineBase,
    PipelineFactory,
)

from ...utils.yaml_dump import deep_convert_dict, merge_comment


class SamplerConfig(BaseModel):
    name: Literal["neighbor"]
    fan_out: List[int] = [5, 10]
    batch_size: int = Field(64, description="Batch size")
    num_workers: int = 4
    eval_batch_size: int = 1024
    eval_num_workers: int = 4

    class Config:
        extra = "forbid"


pipeline_comments = {
    "num_epochs": "Number of training epochs",
    "eval_period": "Interval epochs between evaluations",
    "early_stop": {
        "patience": "Steps before early stop",
        "checkpoint_path": "Early stop checkpoint model file path",
    },
    "sampler": {
        "fan_out": "List of neighbors to sample per edge type for each GNN layer, with the i-th element being the fanout for the i-th GNN layer. Length should be the same as num_layers in model setting",
        "batch_size": "Batch size of seed nodes in training stage",
        "num_workers": "Number of workers to accelerate the graph data processing step",
        "eval_batch_size": "Batch size of seed nodes in training stage in evaluation stage",
        "eval_num_workers": "Number of workers to accelerate the graph data processing step in evaluation stage",
    },
    "save_path": "Directory to save the experiment results",
    "num_runs": "Number of experiments to run",
}


class NodepredNSPipelineCfg(BaseModel):
    sampler: SamplerConfig = Field("neighbor")
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.005, "weight_decay": 0.0}
    loss: str = "CrossEntropyLoss"
    num_runs: int = 1
    save_path: str = "results"


@PipelineFactory.register("nodepred-ns")
class NodepredNsPipeline(PipelineBase):
    def __init__(self):
        self.pipeline = {"name": "nodepred-ns", "mode": "train"}
        self.default_cfg = None

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class NodePredUserConfig(UserConfig):
            eval_device: DeviceEnum = Field("cpu")
            data: DataFactory.filter(
                "nodepred-ns"
            ).get_pydantic_config() = Field(..., discriminator="name")
            model: NodeModelFactory.filter(
                lambda cls: hasattr(cls, "forward_block")
            ).get_pydantic_model_config() = Field(..., discriminator="name")
            general_pipeline: NodepredNSPipelineCfg

        cls.user_cfg_cls = NodePredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter(
                "nodepred-ns"
            ).get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: Optional[str] = typer.Option(
                None, help="output configuration path"
            ),
            model: NodeModelFactory.filter(
                lambda cls: hasattr(cls, "forward_block")
            ).get_model_enum() = typer.Option(..., help="Model name"),
        ):
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline["name"],
                "pipeline_mode": self.pipeline["mode"],
                "device": "cpu",
                "data": {"name": data.name},
                "model": {"name": model.value},
                "general_pipeline": {"sampler": {"name": "neighbor"}},
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            comment_dict = {
                "device": "Torch device name, e.g., cpu or cuda or cuda:0",
                "data": {
                    "split_ratio": "Ratio to generate split masks, for example set to [0.8, 0.1, 0.1] for 80% train/10% val/10% test. Leave blank to use builtin split in original dataset"
                },
                "general_pipeline": pipeline_comments,
                "model": NodeModelFactory.get_constructor_doc_dict(model.value),
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            # truncate length fan_out to be the same as num_layers in model
            if "num_layers" in comment_dict["model"]:
                comment_dict["general_pipeline"]["sampler"]["fan_out"] = [
                    5,
                    10,
                    15,
                    15,
                    15,
                ][: int(comment_dict["model"]["num_layers"])]

            if cfg is None:
                cfg = (
                    "_".join(["nodepred-ns", data.value, model.value]) + ".yaml"
                )
            yaml = ruamel.yaml.YAML()
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print(
                "Configuration file is generated at {}".format(
                    Path(cfg).absolute()
                )
            )

        return config

    @staticmethod
    def gen_script(user_cfg_dict):
        file_current_dir = Path(__file__).resolve().parent
        template_filename = file_current_dir / "nodepred-ns.jinja-py"
        with open(template_filename, "r") as f:
            template = Template(f.read())
        pipeline_cfg = NodepredNSPipelineCfg(
            **user_cfg_dict["general_pipeline"]
        )

        if "num_layers" in user_cfg_dict["model"]:
            assert user_cfg_dict["model"]["num_layers"] == len(
                user_cfg_dict["general_pipeline"]["sampler"]["fan_out"]
            ), "The num_layers in model config should be the same as the length of fan_out in sampler. For example, if num_layers is 1, the fan_out cannot be [5, 10]"

        render_cfg = copy.deepcopy(user_cfg_dict)
        model_code = NodeModelFactory.get_source_code(
            user_cfg_dict["model"]["name"]
        )
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = NodeModelFactory.get_model_class_name(
            user_cfg_dict["model"]["name"]
        )
        render_cfg.update(
            DataFactory.get_generated_code_dict(
                user_cfg_dict["data"]["name"], '**cfg["data"]'
            )
        )
        generated_user_cfg = copy.deepcopy(user_cfg_dict)

        if "split_ratio" in generated_user_cfg["data"]:
            generated_user_cfg["data"].pop("split_ratio")
        generated_user_cfg["data_name"] = generated_user_cfg["data"].pop("name")
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg.pop("pipeline_mode")
        generated_user_cfg["model_name"] = generated_user_cfg["model"].pop(
            "name"
        )
        generated_user_cfg["general_pipeline"]["optimizer"].pop("name")

        if user_cfg_dict["data"].get("split_ratio", None) is not None:
            render_cfg["data_initialize_code"] = "{}, split_ratio={}".format(
                render_cfg["data_initialize_code"],
                user_cfg_dict["data"]["split_ratio"],
            )

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        with open("output.py", "w") as f:
            return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification neighbor sampling pipeline for training"
