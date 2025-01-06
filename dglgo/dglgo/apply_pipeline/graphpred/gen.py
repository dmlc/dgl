from copy import deepcopy
from pathlib import Path
from typing import Optional

import ruamel.yaml
import torch
import typer
from jinja2 import Template
from pydantic import BaseModel, Field

from ...utils.factory import (
    ApplyPipelineFactory,
    DataFactory,
    GraphModelFactory,
    PipelineBase,
)
from ...utils.yaml_dump import deep_convert_dict, merge_comment

pipeline_comments = {
    "batch_size": "Graph batch size",
    "num_workers": "Number of workers for data loading",
    "save_path": "Directory to save the inference results",
}


class ApplyGraphpredPipelineCfg(BaseModel):
    batch_size: int = 32
    num_workers: int = 4
    save_path: str = "apply_results"


@ApplyPipelineFactory.register("graphpred")
class ApplyGraphpredPipeline(PipelineBase):
    def __init__(self):
        self.pipeline = {"name": "graphpred", "mode": "apply"}

    @classmethod
    def setup_user_cfg_cls(cls):
        from ...utils.enter_config import UserConfig

        class ApplyGraphPredUserConfig(UserConfig):
            data: DataFactory.filter("graphpred").get_pydantic_config() = Field(
                ..., discriminator="name"
            )
            general_pipeline: ApplyGraphpredPipelineCfg = (
                ApplyGraphpredPipelineCfg()
            )

        cls.user_cfg_cls = ApplyGraphPredUserConfig

    @property
    def user_cfg_cls(self):
        return self.__class__.user_cfg_cls

    def get_cfg_func(self):
        def config(
            data: DataFactory.filter(
                "graphpred"
            ).get_dataset_enum() = typer.Option(None, help="input data name"),
            cfg: Optional[str] = typer.Option(
                None, help="output configuration file path"
            ),
            cpt: str = typer.Option(..., help="input checkpoint file path"),
        ):
            # Training configuration
            train_cfg = torch.load(cpt, weights_only=False)["cfg"]
            if data is None:
                print("data is not specified, use the training dataset")
                data = train_cfg["data_name"]
            else:
                data = data.name
            if cfg is None:
                cfg = (
                    "_".join(
                        ["apply", "graphpred", data, train_cfg["model_name"]]
                    )
                    + ".yaml"
                )

            self.__class__.setup_user_cfg_cls()
            generated_cfg = {
                "pipeline_name": self.pipeline["name"],
                "pipeline_mode": self.pipeline["mode"],
                "device": train_cfg["device"],
                "data": {"name": data},
                "cpt_path": cpt,
                "general_pipeline": {
                    "batch_size": train_cfg["general_pipeline"][
                        "eval_batch_size"
                    ],
                    "num_workers": train_cfg["general_pipeline"]["num_workers"],
                },
            }
            output_cfg = self.user_cfg_cls(**generated_cfg).dict()
            output_cfg = deep_convert_dict(output_cfg)
            # Not applicable for inference
            output_cfg["data"].pop("split_ratio")
            comment_dict = {
                "device": "Torch device name, e.g., cpu or cuda or cuda:0",
                "cpt_path": "Path to the checkpoint file",
                "general_pipeline": pipeline_comments,
            }
            comment_dict = merge_comment(output_cfg, comment_dict)

            yaml = ruamel.yaml.YAML()
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
        cls.user_cfg_cls(**user_cfg_dict)

        # Training configuration
        train_cfg = torch.load(user_cfg_dict["cpt_path"], weights_only=False)[
            "cfg"
        ]

        # Dict for code rendering
        render_cfg = deepcopy(user_cfg_dict)
        model_name = train_cfg["model_name"]
        model_code = GraphModelFactory.get_source_code(model_name)
        render_cfg["model_code"] = model_code
        render_cfg["model_class_name"] = GraphModelFactory.get_model_class_name(
            model_name
        )
        render_cfg.update(
            DataFactory.get_generated_code_dict(user_cfg_dict["data"]["name"])
        )

        # Dict for defining cfg in the rendered code
        generated_user_cfg = deepcopy(user_cfg_dict)
        generated_user_cfg.pop("pipeline_name")
        generated_user_cfg.pop("pipeline_mode")
        # model arch configuration
        generated_user_cfg["model"] = train_cfg["model"]

        render_cfg["user_cfg_str"] = f"cfg = {str(generated_user_cfg)}"
        render_cfg["user_cfg"] = user_cfg_dict

        file_current_dir = Path(__file__).resolve().parent
        with open(file_current_dir / "graphpred.jinja-py", "r") as f:
            template = Template(f.read())

        return template.render(**render_cfg)

    @staticmethod
    def get_description() -> str:
        return "Graph classification pipeline for inference on binary classification"
