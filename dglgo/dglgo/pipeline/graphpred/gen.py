import copy
from pathlib import Path
from typing import Optional

import ruamel.yaml
import typer
from jinja2 import Template
from pydantic import BaseModel, Field

from ...utils.factory import (
    DataFactory,
    GraphModelFactory,
    PipelineBase,
    PipelineFactory,
)
from ...utils.yaml_dump import deep_convert_dict, merge_comment

pipeline_comments = {
    "num_runs": "Number of experiments to run",
    "train_batch_size": "Graph batch size when training",
    "eval_batch_size": "Graph batch size when evaluating",
    "num_workers": "Number of workers for data loading",
    "num_epochs": "Number of training epochs",
    "save_path": "Directory to save the experiment results",
}


class GraphpredPipelineCfg(BaseModel):
    num_runs: int = 1
    train_batch_size: int = 32
    eval_batch_size: int = 32
    num_workers: int = 4
    optimizer: dict = {"name": "Adam", "lr": 0.001, "weight_decay": 0}
    # Default to no lr decay
    lr_scheduler: dict = {"name": "StepLR", "step_size": 100, "gamma": 1}
    loss: str = "BCEWithLogitsLoss"
    metric: str = "roc_auc_score"
    num_epochs: int = 100
    save_path: str = "results"


@PipelineFactory.register("graphpred")
class GraphpredPipeline(PipelineBase):
    def __init__(self):
        self.pipeline = {"name": "graphpred", "mode": "train"}

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class GraphPredUserConfig(UserConfig):
            data: DataFactory.filter("graphpred").get_pydantic_config() = Field(
                ..., discriminator="name"
            )
            model: GraphModelFactory.get_pydantic_model_config() = Field(
                ..., discriminator="name"
            )
            general_pipeline: GraphpredPipelineCfg = GraphpredPipelineCfg()

        cls.user_cfg_cls = GraphPredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter(
                "graphpred"
            ).get_dataset_enum() = typer.Option(..., help="input data name"),
            cfg: Optional[str] = typer.Option(
                None, help="output configuration path"
            ),
            model: GraphModelFactory.get_model_enum() = typer.Option(
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
                    "split_ratio": "Ratio to generate data split, for example set to [0.8, 0.1, 0.1] for 80% train/10% val/10% test. Leave blank to use builtin split in original dataset"
                },
                "general_pipeline": pipeline_comments,
                "model": GraphModelFactory.get_constructor_doc_dict(
                    model.value
                ),
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            yaml = ruamel.yaml.YAML()
            if cfg is None:
                cfg = "_".join(["graphpred", data.value, model.value]) + ".yaml"
            yaml.dump(comment_dict, Path(cfg).open("w"))
            print(
                "Configuration file is generated at {}".format(
                    Path(cfg).absolute()
                )
            )

        return config

    @classmethod
    def gen_script(cls, user_cfg_dict):
        cls.setup_user_cfg_cls()
        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "graphpred.jinja-py", "r") as f:
            template = Template(f.read())

        render_cfg = copy.deepcopy(user_cfg_dict)
        model_code = GraphModelFactory.get_source_code(
            user_cfg_dict["model"]["name"]
        )
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = GraphModelFactory.get_model_class_name(
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
        generated_user_cfg["general_pipeline"]["lr_scheduler"].pop("name")

        generated_train_cfg = copy.deepcopy(user_cfg_dict["general_pipeline"])
        generated_train_cfg["optimizer"].pop("name")
        generated_train_cfg["lr_scheduler"].pop("name")

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
        return "Graph property prediction pipeline on binary classification"
