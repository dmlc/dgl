import copy
from pathlib import Path
from typing import Optional

import ruamel.yaml
import typer
import yaml
from jinja2 import Template
from pydantic import BaseModel, Field
from ruamel.yaml.comments import CommentedMap

from ...utils.base_model import DeviceEnum, EarlyStopConfig
from ...utils.factory import (
    DataFactory,
    NodeModelFactory,
    PipelineBase,
    PipelineFactory,
)
from ...utils.yaml_dump import deep_convert_dict, merge_comment

pipeline_comments = {
    "num_epochs": "Number of training epochs",
    "eval_period": "Interval epochs between evaluations",
    "early_stop": {
        "patience": "Steps before early stop",
        "checkpoint_path": "Early stop checkpoint model file path",
    },
    "save_path": "Directory to save the experiment results",
    "num_runs": "Number of experiments to run",
}


class NodepredPipelineCfg(BaseModel):
    early_stop: Optional[EarlyStopConfig] = EarlyStopConfig()
    num_epochs: int = 200
    eval_period: int = 5
    optimizer: dict = {"name": "Adam", "lr": 0.01, "weight_decay": 5e-4}
    loss: str = "CrossEntropyLoss"
    save_path: str = "results"
    num_runs: int = 1


@PipelineFactory.register("nodepred")
class NodepredPipeline(PipelineBase):

    user_cfg_cls = None

    def __init__(self):
        self.pipeline = {"name": "nodepred", "mode": "train"}

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class NodePredUserConfig(UserConfig):
            data: DataFactory.filter("nodepred").get_pydantic_config() = Field(
                ..., discriminator="name"
            )
            model: NodeModelFactory.get_pydantic_model_config() = Field(
                ..., discriminator="name"
            )
            general_pipeline: NodepredPipelineCfg = NodepredPipelineCfg()

        cls.user_cfg_cls = NodePredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter(
                "nodepred"
            ).get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: Optional[str] = typer.Option(
                None, help="output configuration path"
            ),
            model: NodeModelFactory.get_model_enum() = typer.Option(
                ..., help="Model name"
            ),
        ):
            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline["name"],
                "pipeline_mode": self.pipeline["mode"],
                "device": "cpu",
                "data": {"name": data.name},
                "model": {"name": model.value},
                "general_pipeline": {},
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

            yaml = ruamel.yaml.YAML()
            if cfg is None:
                cfg = "_".join(["nodepred", data.value, model.value]) + ".yaml"
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print(
                "Configuration file is generated at {}".format(
                    Path(cfg).absolute()
                )
            )

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        # Check validation
        cls.setup_user_cfg_cls()
        user_cfg = cls.user_cfg_cls(**user_cfg_dict)
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "nodepred.jinja-py", "r") as f:
            template = Template(f.read())

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

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")

        if user_cfg_dict["data"].get("split_ratio", None) is not None:
            render_cfg["data_initialize_code"] = "{}, split_ratio={}".format(
                render_cfg["data_initialize_code"],
                user_cfg_dict["data"]["split_ratio"],
            )
        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict
        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Node classification pipeline for training"
